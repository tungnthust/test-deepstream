/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __GST_NVDSIMAGESAVER_H__
#define __GST_NVDSIMAGESAVER_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
// DeepStream headers
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"
#include <nvll_osd_api.h> 

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

// --- Forward declarations ---
typedef struct _GHashTable GHashTable;
typedef struct _GQueue GQueue;
typedef struct _GString GString;
typedef struct _GThread GThread;

// --- Structures matching the provided file, enhanced for streams ---
#define VIDEO_RECORD_FRAMES 75

typedef struct {
    guint width;
    guint height;
    NvBufSurfaceColorFormat format;
    guint frame_number;
    GstClockTime timestamp;
    guint8 *data;
    gsize data_size;
} BufferedFrame;

typedef struct {
    guint session_id;
    gint stream_id; // --- KEY ADDITION: Track originating stream ---
    guint start_frame;
    GString *object_ids;
    GQueue *frames; // Queue of BufferedFrame*
    guint frames_recorded;
    gboolean is_complete;
    gchar *output_filename;
} RecordingSession;

typedef struct {
    RecordingSession *session;
    GstNvDsImageSaver *saver; // Reference back to the plugin instance
} VideoEncodeWorkItem;

struct _GstNvDsImageSaver
{
    GstBaseTransform base_trans;

    // Properties (matching provided file)
    guint unique_id;
    gchar *output_path;
    gboolean enable;
    gboolean save_full_frame;
    gboolean save_crops;
    gboolean save_each_object_once;
    gboolean enable_video_recording; // New property from provided file

    guint frame_counter;
    guint next_session_id; // For generating unique session IDs

    // --- Image Saving State (matching provided file) ---
    GHashTable *stream_saved_objects; // Key: gint* (stream_id), Value: GHashTable* (set of guint64 object_ids)

    // --- Video Recording State (matching provided file, enhanced for streams) ---
    // Key: gint (stream_id), Value: GQueue* (per-stream buffer of BufferedFrame*)
    GHashTable *stream_buffers;
    // Key: guint (session_id), Value: RecordingSession*
    GHashTable *active_recordings;
    guint max_buffer_frames; // Size of per-stream buffer

    // Threading for encoding (matching provided file)
    GQueue *encoding_queue;
    GMutex encoding_mutex;
    GCond encoding_cond;
    gboolean encoding_thread_running;
    GThread *encoding_thread;

    // Mutex for buffer access
    GMutex buffer_mutex; // Mutex for accessing stream_buffers
};

struct _GstNvDsImageSaverClass
{
    GstBaseTransformClass parent_class;
};

GType gst_nvdsimagesaver_get_type (void);

G_END_DECLS

#endif /* __GST_NVDSIMAGESAVER_H__ */