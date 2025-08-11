// bbox_draw.cpp - GStreamer/DeepStream integration with debugging
#include "bbox_draw.h"
#include "bbox_draw_kernels.cuh"
#include <gst/gst.h>
#include <cuda_runtime.h>

gboolean draw_bounding_boxes_cuda_kernel(NvBufSurface *surface, NvDsFrameMeta *frame_meta, guint batch_id) {
    GST_INFO("🔍 draw_bounding_boxes_cuda_kernel called with batch_id=%u", batch_id);
    
    // Validate inputs
    if (!surface || !frame_meta || batch_id >= surface->numFilled) {
        GST_WARNING("❌ Invalid input for CUDA kernel drawing: surface=%p, frame_meta=%p, batch_id=%u, numFilled=%u",
                   surface, frame_meta, batch_id, surface ? surface->numFilled : 0);
        return FALSE;
    }

    GST_INFO("✅ Input validation passed");

    // Check memory type
    if (surface->memType != NVBUF_MEM_CUDA_DEVICE) {
         GST_WARNING("❌ Surface memory type (%d) is not CUDA device (%d). CUDA kernel requires CUDA device memory.", 
                    surface->memType, NVBUF_MEM_CUDA_DEVICE);
         return FALSE;
    }

    GST_INFO("✅ Memory type is CUDA device");

    // Get surface parameters for the specific batch
    NvBufSurfaceParams *params = &surface->surfaceList[batch_id];

    GST_INFO("📊 Surface info: width=%u, height=%u, pitch=%u, colorFormat=%d", 
             params->width, params->height, params->pitch, params->colorFormat);

    // Check format - this kernel is written for RGBA
    if (params->colorFormat != NVBUF_COLOR_FORMAT_RGBA) {
        GST_WARNING("❌ CUDA kernel drawing expects RGBA format (%d), got %d.", 
                   NVBUF_COLOR_FORMAT_RGBA, params->colorFormat);
        
        // Let's be more flexible and try other formats
        if (params->colorFormat == NVBUF_COLOR_FORMAT_NV12 || 
            params->colorFormat == NVBUF_COLOR_FORMAT_NV12_ER ||
            params->colorFormat == NVBUF_COLOR_FORMAT_NV12_709_ER ||
            params->colorFormat == NVBUF_COLOR_FORMAT_BGRA) {
            GST_WARNING("⚠️  Continuing with format %d (may not work correctly)", params->colorFormat);
        } else {
            return FALSE;
        }
    }

    // Check for valid data pointer
    if (!params->dataPtr) {
        GST_ERROR("❌ Surface data pointer is null for batch_id %u.", batch_id);
        return FALSE;
    }

    GST_INFO("✅ Surface data pointer is valid: %p", params->dataPtr);

    gboolean drawn_any = FALSE;
    cudaStream_t stream = 0; // Use default stream

    // Count total objects first
    int obj_count = 0;
    NvDsMetaList *l_obj_count = NULL;
    for (l_obj_count = frame_meta->obj_meta_list; l_obj_count != NULL; l_obj_count = l_obj_count->next) {
        obj_count++;
    }
    
    GST_INFO("📦 Found %d objects in frame metadata", obj_count);

    // Iterate through objects in the frame meta
    NvDsMetaList *l_obj = NULL;
    int processed_objects = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
        NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
        processed_objects++;
        
        if (!obj_meta) {
            GST_WARNING("⚠️  Object %d: obj_meta is NULL", processed_objects);
            continue;
        }

        GST_INFO("🎯 Object %d: id=%lu, class_id=%d, rect=[%.2f, %.2f, %.2f, %.2f]",
                 processed_objects, obj_meta->object_id, obj_meta->class_id,
                 obj_meta->rect_params.left, obj_meta->rect_params.top,
                 obj_meta->rect_params.width, obj_meta->rect_params.height);

        if (obj_meta->rect_params.width <= 0 || obj_meta->rect_params.height <= 0) {
            GST_WARNING("⚠️  Object %d: Invalid dimensions (w=%.2f, h=%.2f)", 
                       processed_objects, obj_meta->rect_params.width, obj_meta->rect_params.height);
            continue;
        }

        // Check if rectangle is within image bounds
        if (obj_meta->rect_params.left < 0 || obj_meta->rect_params.top < 0 ||
            obj_meta->rect_params.left + obj_meta->rect_params.width > params->width ||
            obj_meta->rect_params.top + obj_meta->rect_params.height > params->height) {
            GST_WARNING("⚠️  Object %d: Rectangle extends outside image bounds. Image: %ux%u, Rect: [%.2f, %.2f, %.2f, %.2f]",
                       processed_objects, params->width, params->height,
                       obj_meta->rect_params.left, obj_meta->rect_params.top,
                       obj_meta->rect_params.width, obj_meta->rect_params.height);
            // Continue anyway, kernel should handle clipping
        }

        // Define drawing parameters
        const int border_width = 4;
        const unsigned char r = 255, g = 0, b = 0; // Red

        GST_INFO("🚀 Launching kernel for object %d...", processed_objects);

        // Launch the kernel via the C wrapper
        launch_draw_rectangle_kernel(
            stream,
            (unsigned char*)params->dataPtr,
            (int)params->width, (int)params->height, (int)params->pitch,
            (int)obj_meta->rect_params.left, (int)obj_meta->rect_params.top,
            (int)obj_meta->rect_params.width, (int)obj_meta->rect_params.height,
            border_width, r, g, b
        );

        // Check for kernel launch errors immediately
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            GST_ERROR("❌ CUDA kernel launch failed for object %d (id=%lu): %s", 
                     processed_objects, obj_meta->object_id, cudaGetErrorString(err));
        } else {
            GST_INFO("✅ Kernel launched successfully for object %d", processed_objects);
            drawn_any = TRUE;
        }
    }

    GST_INFO("🔄 Processed %d/%d objects, drawn_any=%s", processed_objects, obj_count, drawn_any ? "TRUE" : "FALSE");

    // Synchronize to ensure all drawing is complete
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        GST_ERROR("❌ cudaStreamSynchronize after drawing failed: %s", cudaGetErrorString(sync_err));
        return FALSE;
    } else {
        GST_INFO("✅ CUDA synchronization successful");
        if (drawn_any) {
             GST_INFO("🎉 Successfully drew bounding boxes using CUDA kernel for batch_id %u.", batch_id);
        } else {
             GST_INFO("ℹ️  No boxes drawn using CUDA kernel for batch_id %u (no valid objects found).", batch_id);
        }
    }

    return drawn_any;
}