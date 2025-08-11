// bbox_draw_kernels.cu - Pure CUDA implementation with debugging
#include "bbox_draw_kernels.cuh"
#include <stdio.h>

// CUDA kernel implementation with debugging
__global__ void draw_rectangle_kernel(unsigned char* data, int width, int height, int pitch,
                                     int left, int top, int rect_width, int rect_height,
                                     int border_width, unsigned char r, unsigned char g, unsigned char b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (x >= width || y >= height) return;

    // Check if pixel is on the border of the rectangle
    bool on_border = false;
    
    // Check if pixel is inside the rectangle bounds
    if (x >= left && x < left + rect_width && y >= top && y < top + rect_height) {
        // Check if pixel is within the border width from any edge
        if (x < left + border_width || x >= left + rect_width - border_width ||
            y < top + border_width || y >= top + rect_height - border_width) {
            on_border = true;
        }
    }

    if (on_border) {
        // Calculate pixel index - assuming 4 bytes per pixel (RGBA)
        int idx = y * pitch + x * 4;
        
        // Bounds check for the buffer
        if (idx >= 0 && idx + 2 < height * pitch) {
            // Set RGB values (assuming RGBA format)
            data[idx] = r;     // R
            data[idx + 1] = g; // G
            data[idx + 2] = b; // B
            // Note: We're not modifying the alpha channel (idx + 3)
            
            // Debug: Print first few pixels that get modified (only from thread 0,0 to avoid spam)
            if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
                printf("CUDA: Drawing pixel at (%d,%d), idx=%d, color=(%d,%d,%d)\n", x, y, idx, r, g, b);
            }
        }
    }
}

// C wrapper function with debugging
extern "C" {
void launch_draw_rectangle_kernel(cudaStream_t stream,
                                 unsigned char* d_data, 
                                 int width, int height, int pitch,
                                 int left, int top, int rect_width, int rect_height,
                                 int border_width, 
                                 unsigned char r, unsigned char g, unsigned char b) {
    
    // Debug print kernel parameters
    printf("CUDA Kernel Launch Parameters:\n");
    printf("  Surface: %dx%d, pitch=%d\n", width, height, pitch);
    printf("  Rectangle: [%d,%d] %dx%d\n", left, top, rect_width, rect_height);
    printf("  Border width: %d\n", border_width);
    printf("  Color: RGB(%d,%d,%d)\n", r, g, b);
    printf("  Data pointer: %p\n", d_data);

    // Validate parameters
    if (!d_data) {
        printf("ERROR: Data pointer is NULL!\n");
        return;
    }
    
    if (width <= 0 || height <= 0 || pitch <= 0) {
        printf("ERROR: Invalid surface dimensions!\n");
        return;
    }
    
    if (rect_width <= 0 || rect_height <= 0) {
        printf("ERROR: Invalid rectangle dimensions!\n");
        return;
    }

    // Calculate grid and block dimensions
    const int block_size = 16;
    dim3 blockSize(block_size, block_size);
    dim3 gridSize((width + block_size - 1) / block_size,
                  (height + block_size - 1) / block_size);

    printf("  Grid size: %dx%d, Block size: %dx%d\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);

    // Launch the kernel
    draw_rectangle_kernel<<<gridSize, blockSize, 0, stream>>>(
        d_data, width, height, pitch,
        left, top, rect_width, rect_height,
        border_width, r, g, b
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Kernel launch failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("SUCCESS: Kernel launched successfully\n");
    }
}
}