// bbox_draw_kernels.cuh - CUDA kernel declarations (no GLib headers)
#pragma once
#include <cuda_runtime.h>

// CUDA kernel for drawing rectangles
__global__ void draw_rectangle_kernel(unsigned char* data, int width, int height, int pitch,
                                     int left, int top, int rect_width, int rect_height,
                                     int border_width, unsigned char r, unsigned char g, unsigned char b);

// C wrapper function (callable from C++ code that includes GLib headers)
extern "C" {
void launch_draw_rectangle_kernel(cudaStream_t stream,
                                 unsigned char* d_data, 
                                 int width, int height, int pitch,
                                 int left, int top, int rect_width, int rect_height,
                                 int border_width, 
                                 unsigned char r, unsigned char g, unsigned char b);
}