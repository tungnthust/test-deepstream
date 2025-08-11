/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __GST_NVDSIMAGESAVER_H__
#define __GST_NVDSIMAGESAVER_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
// Removed unused <vector>, <string>

// DeepStream headers
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"

/* Package and library details required for plugin_init */
#define PACKAGE "nvdsimagesaver"
#define VERSION "1.0"
#define LICENSE "MIT"
#define DESCRIPTION "NVIDIA dsimagesaver plugin to save images and update NvDsEventMsgMeta with file paths"
#define BINARY_PACKAGE "NVIDIA DeepStream dsimagesaver plugin"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS

typedef struct _GstNvDsImageSaver GstNvDsImageSaver;
typedef struct _GstNvDsImageSaverClass GstNvDsImageSaverClass;

/* Standard GType stuff */
#define GST_TYPE_NVDSIMAGESAVER (gst_nvdsimagesaver_get_type())
#define GST_NVDSIMAGESAVER(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSIMAGESAVER,GstNvDsImageSaver))
#define GST_NVDSIMAGESAVER_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSIMAGESAVER,GstNvDsImageSaverClass))
#define GST_IS_NVDSIMAGESAVER(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSIMAGESAVER))
#define GST_IS_NVDSIMAGESAVER_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSIMAGESAVER))
#define GST_NVDSIMAGESAVER_CAST(obj)  ((GstNvDsImageSaver *)(obj))

// --- CHANGED: Forward declaration for GHashTable ---
typedef struct _GHashTable GHashTable;

struct _GstNvDsImageSaver
{
    GstBaseTransform base_trans;

    // Properties
    guint unique_id;              // Unique ID for the element instance
    gchar *output_path;           // Base path for saved images (e.g., "/tmp/saved_frames/")
    gboolean enable;              // Enable/disable saving
    gboolean save_full_frame;     // Enable/disable full frame saving
    gboolean save_crops;          // Enable/disable cropped object saving
    gboolean save_each_object_once; // Enable/disable saving each object only once PER STREAM
    guint frame_counter;          // Counter to generate unique filenames

    // --- CHANGED: Track saved object IDs per stream ---
    // Key: stream_id (gint*), Value: GHashTable* (set of guint64 object_ids for that stream)
    GHashTable *stream_saved_objects;
};

struct _GstNvDsImageSaverClass
{
    GstBaseTransformClass parent_class;
};

GType gst_nvdsimagesaver_get_type (void);

G_END_DECLS

#endif /* __GST_NVDSIMAGESAVER_H__ */