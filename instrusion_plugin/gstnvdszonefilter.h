/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __GST_NVDSZONEFILTER_H__
#define __GST_NVDSZONEFILTER_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory> // For std::unique_ptr if needed

// DeepStream headers
#include "nvbufsurface.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"

/* Package and library details required for plugin_init */
#define PACKAGE "nvdszonefilter"
#define VERSION "1.0"
#define LICENSE "MIT" // Or appropriate license
#define DESCRIPTION "NVIDIA dszonefilter plugin for filtering objects based on zones"
#define BINARY_PACKAGE "NVIDIA DeepStream dszonefilter plugin"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS

typedef struct _GstNvDsZoneFilter GstNvDsZoneFilter;
typedef struct _GstNvDsZoneFilterClass GstNvDsZoneFilterClass;

/* Standard GType stuff */
#define GST_TYPE_NVDSZONEFILTER (gst_nvdszonefilter_get_type())
#define GST_NVDSZONEFILTER(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSZONEFILTER,GstNvDsZoneFilter))
#define GST_NVDSZONEFILTER_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSZONEFILTER,GstNvDsZoneFilterClass))
#define GST_IS_NVDSZONEFILTER(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSZONEFILTER))
#define GST_IS_NVDSZONEFILTER_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSZONEFILTER))
#define GST_NVDSZONEFILTER_CAST(obj) ((GstNvDsZoneFilter *)(obj))
extern GstDebugCategory *gst_nvdszonefilter_debug;



// --- Structure to hold parsed zone data for a stream ---
typedef struct {
    gint stream_id;
    gdouble x1, y1, x2, y2; // Normalized coordinates
    gboolean valid; // Flag to indicate if the stream config was parsed successfully
} StreamZoneConfig;

struct _GstNvDsZoneFilter {
    GstBaseTransform base_trans;

    // Unique ID for the element
    guint unique_id;

    // Config file path
    gchar *config_file_path;

    // Config file parsing status
    gboolean config_file_parse_successful;

    // Mutex for thread safety during config access
    GMutex config_mutex;

    // Map to store zone configuration per stream ID
    // Key: stream_id (gint), Value: StreamZoneConfig
    std::unordered_map<gint, StreamZoneConfig> *stream_zones_map;

    // Enable flag (passthrough if disabled)
    gboolean enable;
};

struct _GstNvDsZoneFilterClass {
    GstBaseTransformClass parent_class;
};

GType gst_nvdszonefilter_get_type(void);

G_END_DECLS
#endif /* __GST_NVDSZONEFILTER_H__ */