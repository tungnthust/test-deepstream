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
// Removed unused includes: <iostream>, <vector>, <memory>
#include <unordered_map>
#include <unordered_set> // Added for std::unordered_set
#include <string>
#include <vector> // Added for std::vector (used in transform_ip for ids_to_remove)
// #include <string> is usually included by other headers, but good to be explicit if needed.

// DeepStream headers
#include "nvbufsurface.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"

/* Package and library details required for plugin_init */
// These are often defined by the build system (e.g., CMake) via config.h or similar.
// If not, defining them here is a common fallback.
#ifndef PACKAGE
#define PACKAGE "nvdszonefilter"
#endif
#ifndef VERSION
#define VERSION "1.0"
#endif
#ifndef LICENSE
#define LICENSE "MIT"
#endif
#ifndef DESCRIPTION
#define DESCRIPTION "NVIDIA dszonefilter plugin for filtering objects based on zones using NvDsEventMsgMeta (Entry only once with debouncing and zone drawing)"
#endif
#ifndef BINARY_PACKAGE
#define BINARY_PACKAGE "NVIDIA DeepStream dszonefilter plugin"
#endif
#ifndef URL
#define URL "http://nvidia.com/"
#endif


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

// Debug category declaration - defined in the .cpp file
extern GstDebugCategory *gst_nvdszonefilter_debug;

// --- Structure to hold parsed zone data for a stream ---
typedef struct {
    gint stream_id;
    gdouble x1, y1, x2, y2; // Normalized coordinates
    gboolean valid; // Flag to indicate if the stream config was parsed successfully
} StreamZoneConfig;

// --- Structure to hold tracking information for an object in a zone ---
typedef struct {
    guint64 last_seen_frame_num; // Frame number when the object was last seen in the zone
    // Add other state if needed (e.g., confidence history, position history)
} TrackedObjectInfo;

struct _GstNvDsZoneFilter {
    GstBaseTransform base_trans;

    // Unique ID for the element
    guint unique_id;

    // Config file path
    gchar *config_file_path;

    // Config file parsing status
    gboolean config_file_parse_successful;

    // Mutex for thread safety during config access and tracked object state
    GMutex config_mutex;

    // Map to store zone configuration per stream ID
    // Key: stream_id (gint), Value: StreamZoneConfig
    std::unordered_map<gint, StreamZoneConfig> *stream_zones_map;

    // Map to track object IDs currently inside or recently inside the zone for each stream
    // Key: stream_id (gint), Value: Map of object_id (guint64) -> TrackedObjectInfo
    std::unordered_map<gint, std::unordered_map<guint64, TrackedObjectInfo>> *tracked_objects_info;

    // Debounce threshold in frames
    guint debounce_frame_count;

    // Enable flag (passthrough if disabled)
    gboolean enable;

    // Zone drawing flag (enable/disable zone overlay drawing)
    gboolean draw_zones;
};

struct _GstNvDsZoneFilterClass
{
  GstBaseTransformClass parent_class;
};

GType gst_nvdszonefilter_get_type (void);

G_END_DECLS
#endif /* __GST_NVDSZONEFILTER_H__ */