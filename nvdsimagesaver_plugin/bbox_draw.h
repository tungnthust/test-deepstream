#ifndef BBOX_DRAW_H
#define BBOX_DRAW_H

#include <nvbufsurface.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"
#include <glib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Function to draw bounding boxes using CUDA kernel
gboolean draw_bounding_boxes_cuda_kernel(NvBufSurface *surface, NvDsFrameMeta *frame_meta, guint batch_id);

#ifdef __cplusplus
}
#endif

#endif // BBOX_DRAW_H