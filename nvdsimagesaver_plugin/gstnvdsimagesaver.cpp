/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */


#include "gstnvdsimagesaver.h"
#include <sys/stat.h> // For mkdir
#include <unistd.h>   // For access
#include <cstdio>     // For snprintf
#include <opencv2/opencv.hpp> // For image/video encoding
#include <cuda_runtime.h>
#include <string.h>

GST_DEBUG_CATEGORY_STATIC (gst_nvdsimagesaver_debug);
#define GST_CAT_DEFAULT gst_nvdsimagesaver_debug

/* Enum to identify properties (matching provided file) */
enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_OUTPUT_PATH,
  PROP_ENABLE,
  PROP_SAVE_FULL_FRAME,
  PROP_SAVE_CROPS,
  PROP_SAVE_EACH_OBJECT_ONCE,
  PROP_ENABLE_VIDEO_RECORDING // New property
};

/* Default values for properties (matching provided file) */
#define DEFAULT_UNIQUE_ID 20
#define DEFAULT_OUTPUT_PATH "/tmp/saved_frames/"
#define DEFAULT_ENABLE TRUE
#define DEFAULT_SAVE_FULL_FRAME TRUE
#define DEFAULT_SAVE_CROPS TRUE
#define DEFAULT_SAVE_EACH_OBJECT_ONCE TRUE
#define DEFAULT_ENABLE_VIDEO_RECORDING TRUE

#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA }")));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA }")));

#define gst_nvdsimagesaver_parent_class parent_class
G_DEFINE_TYPE (GstNvDsImageSaver, gst_nvdsimagesaver, GST_TYPE_BASE_TRANSFORM);

/* Function prototypes (matching provided file) */
static void gst_nvdsimagesaver_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvdsimagesaver_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static GstFlowReturn gst_nvdsimagesaver_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf);
static gboolean gst_nvdsimagesaver_set_caps (GstBaseTransform * btrans, GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_nvdsimagesaver_start (GstBaseTransform * btrans);
static gboolean gst_nvdsimagesaver_stop (GstBaseTransform * btrans);
static void gst_nvdsimagesaver_finalize (GObject * object);

// --- Helper functions for GHashTable keys/values (from previous solution) ---
static guint int_hash(gconstpointer key) { return g_int_hash(key); }
static gboolean int_equal(gconstpointer a, gconstpointer b) { return g_int_equal(a, b); }
static guint int64_hash(gconstpointer key) { return g_int64_hash(key); }
static gboolean int64_equal(gconstpointer a, gconstpointer b) { return g_int64_equal(a, b); }
static void destroy_stream_object_set(gpointer data) {
    if (data) g_hash_table_destroy((GHashTable*)data);
}
static void destroy_buffered_frame_queue(gpointer data) {
    if (data) {
        GQueue *q = (GQueue*)data;
        while(!g_queue_is_empty(q)) {
            BufferedFrame *f = (BufferedFrame*)g_queue_pop_head(q);
            if (f) {
                g_free(f->data);
                g_free(f);
            }
        }
        g_queue_free(q);
    }
}
// --- END Helper functions ---

// --- Video recording helper functions (matching provided file signature, enhanced for streams) ---
static BufferedFrame* create_buffered_frame(NvBufSurface *surface, guint batch_id, guint frame_number, GstClockTime timestamp);
static void destroy_buffered_frame(BufferedFrame *frame);
static RecordingSession* create_recording_session(guint session_id, gint stream_id, guint start_frame, const gchar* object_ids, const gchar* output_path);
static void destroy_recording_session(RecordingSession *session);
// --- CHANGED: Stream-aware buffering ---
static void add_frame_to_stream_buffer(GstNvDsImageSaver *saver, gint stream_id, BufferedFrame *frame);
static GQueue* get_or_create_stream_buffer(GstNvDsImageSaver *saver, gint stream_id);
// --- END CHANGED ---
static void start_recording_for_objects(GstNvDsImageSaver *saver, gint stream_id, const gchar* object_ids); // Stream ID passed
static gboolean is_zone_entry_event(NvDsEventMsgMeta *event_meta);
static gchar* extract_object_ids_from_events(NvDsFrameMeta *frame_meta);
static gpointer encoding_thread_func(gpointer user_data);
static void encode_recording_session(RecordingSession *session, GstNvDsImageSaver *saver);

// --- Image saving helper functions (matching provided file) ---
static gboolean save_nvmm_buffer_as_jpeg_crop(NvBufSurface *surface, guint batch_id, const gchar *filename,
                                               gdouble crop_left, gdouble crop_top, gdouble crop_width, gdouble crop_height);
static gboolean ensure_output_directory(const gchar *path);
static gboolean is_object_already_saved(GstNvDsImageSaver *saver, guint64 object_id, gint stream_id);
static void mark_object_as_saved(GstNvDsImageSaver *saver, guint64 object_id, gint stream_id);
static void update_event_meta_with_image_paths(NvDsEventMsgMeta *event_meta,
                                   const gchar *full_frame_path,
                                   const gchar *crop_path,
                                   const gchar *video_path);

static void
gst_nvdsimagesaver_class_init (GstNvDsImageSaverClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  gst_element_class_set_details_simple (gstelement_class,
      "DsImageSaver Plugin",
      "Sink/Video",
      "Saves images/videos and updates NvDsEventMsgMeta with file paths (only if events exist)",
      "Your Name / NVIDIA");

  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_nvdsimagesaver_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_nvdsimagesaver_get_property);
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_nvdsimagesaver_finalize);

  // Install properties (matching provided file)
  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id", "Unique ID",
          "Unique ID for the element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_OUTPUT_PATH,
      g_param_spec_string ("output-path", "Output Path",
          "Base directory path to save images/videos (must end with /)",
          DEFAULT_OUTPUT_PATH, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_ENABLE,
      g_param_spec_boolean ("enable", "Enable",
          "Enable image/video saving",
          DEFAULT_ENABLE, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));
  g_object_class_install_property (gobject_class, PROP_SAVE_FULL_FRAME,
      g_param_spec_boolean ("save-full-frame", "Save Full Frame",
          "Save full frame images",
          DEFAULT_SAVE_FULL_FRAME, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_SAVE_CROPS,
      g_param_spec_boolean ("save-crops", "Save Crops",
          "Save cropped object images",
          DEFAULT_SAVE_CROPS, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_SAVE_EACH_OBJECT_ONCE,
      g_param_spec_boolean ("save-each-object-once", "Save Each Object Once",
          "Save each object only once per stream across all frames",
          DEFAULT_SAVE_EACH_OBJECT_ONCE, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property (gobject_class, PROP_ENABLE_VIDEO_RECORDING,
      g_param_spec_boolean ("enable-video-recording", "Enable Video Recording",
          "Enable video recording on zone entry events",
          DEFAULT_ENABLE_VIDEO_RECORDING, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_nvdsimagesaver_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvdsimagesaver_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvdsimagesaver_stop);
  gstbasetransform_class->transform_ip = GST_DEBUG_FUNCPTR (gst_nvdsimagesaver_transform_ip);

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_template));
}

static void
gst_nvdsimagesaver_init (GstNvDsImageSaver * saver)
{
  GST_DEBUG_OBJECT (saver, "Initializing GstNvDsImageSaver");
  // Initialize properties (matching provided file)
  saver->unique_id = DEFAULT_UNIQUE_ID;
  saver->output_path = g_strdup(DEFAULT_OUTPUT_PATH);
  saver->enable = DEFAULT_ENABLE;
  saver->save_full_frame = DEFAULT_SAVE_FULL_FRAME;
  saver->save_crops = DEFAULT_SAVE_CROPS;
  saver->save_each_object_once = DEFAULT_SAVE_EACH_OBJECT_ONCE;
  saver->enable_video_recording = DEFAULT_ENABLE_VIDEO_RECORDING;
  saver->frame_counter = 0;
  saver->next_session_id = 1;

  // Initialize image saving state (matching provided file)
  saver->stream_saved_objects = g_hash_table_new_full(
      int_hash, int_equal, g_free, destroy_stream_object_set);

  // Initialize video recording state (enhanced for streams)
  saver->stream_buffers = g_hash_table_new_full(
      int_hash, int_equal, g_free, destroy_buffered_frame_queue); // Key: gint*, Value: GQueue*
  saver->active_recordings = g_hash_table_new_full(g_direct_hash, g_direct_equal, NULL, (GDestroyNotify)destroy_recording_session);
  saver->max_buffer_frames = VIDEO_RECORD_FRAMES + 5;
  saver->encoding_queue = g_queue_new();
  g_mutex_init(&saver->encoding_mutex);
  g_cond_init(&saver->encoding_cond);
  g_mutex_init(&saver->buffer_mutex);
  saver->encoding_thread_running = FALSE;
  saver->encoding_thread = NULL;

  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (saver), TRUE);
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (saver), FALSE);
}

static void
gst_nvdsimagesaver_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvDsImageSaver *saver = GST_NVDSIMAGESAVER (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      saver->unique_id = g_value_get_uint (value);
      break;
    case PROP_OUTPUT_PATH: {
      g_free(saver->output_path);
      saver->output_path = g_value_dup_string(value);
      if (saver->output_path && strlen(saver->output_path) > 0 &&
          saver->output_path[strlen(saver->output_path) - 1] != '/') {
          gchar *tmp = g_strconcat(saver->output_path, "/", NULL);
          g_free(saver->output_path);
          saver->output_path = tmp;
      }
      GST_DEBUG_OBJECT(saver, "Set output path to: %s", saver->output_path);
      break;
    }
    case PROP_ENABLE:
      saver->enable = g_value_get_boolean (value);
      break;
    case PROP_SAVE_FULL_FRAME:
      saver->save_full_frame = g_value_get_boolean (value);
      break;
    case PROP_SAVE_CROPS:
      saver->save_crops = g_value_get_boolean (value);
      break;
    case PROP_SAVE_EACH_OBJECT_ONCE:
      saver->save_each_object_once = g_value_get_boolean (value);
      break;
    case PROP_ENABLE_VIDEO_RECORDING:
      saver->enable_video_recording = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_nvdsimagesaver_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvDsImageSaver *saver = GST_NVDSIMAGESAVER (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, saver->unique_id);
      break;
    case PROP_OUTPUT_PATH:
      g_value_set_string (value, saver->output_path);
      break;
    case PROP_ENABLE:
      g_value_set_boolean (value, saver->enable);
      break;
    case PROP_SAVE_FULL_FRAME:
      g_value_set_boolean (value, saver->save_full_frame);
      break;
    case PROP_SAVE_CROPS:
      g_value_set_boolean (value, saver->save_crops);
      break;
    case PROP_SAVE_EACH_OBJECT_ONCE:
      g_value_set_boolean (value, saver->save_each_object_once);
      break;
    case PROP_ENABLE_VIDEO_RECORDING:
      g_value_set_boolean (value, saver->enable_video_recording);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_nvdsimagesaver_finalize (GObject * object)
{
  GstNvDsImageSaver *saver = GST_NVDSIMAGESAVER (object);
  GST_DEBUG_OBJECT (saver, "Finalizing GstNvDsImageSaver");

  // Stop encoding thread (matching provided file)
  if (saver->encoding_thread_running) {
    g_mutex_lock(&saver->encoding_mutex);
    saver->encoding_thread_running = FALSE;
    g_cond_signal(&saver->encoding_cond);
    g_mutex_unlock(&saver->encoding_mutex);
    if (saver->encoding_thread) {
      g_thread_join(saver->encoding_thread);
      saver->encoding_thread = NULL;
    }
  }

  // Clean up stream buffers
  g_mutex_lock(&saver->buffer_mutex);
  if (saver->stream_buffers) {
    g_hash_table_destroy(saver->stream_buffers);
    saver->stream_buffers = NULL;
  }
  g_mutex_unlock(&saver->buffer_mutex);
  g_mutex_clear(&saver->buffer_mutex);

  // Clean up encoding queue (matching provided file)
  g_mutex_lock(&saver->encoding_mutex);
  if (saver->encoding_queue) {
    while (!g_queue_is_empty(saver->encoding_queue)) {
      VideoEncodeWorkItem *item = (VideoEncodeWorkItem*)g_queue_pop_head(saver->encoding_queue);
      g_free(item);
    }
    g_queue_free(saver->encoding_queue);
    saver->encoding_queue = NULL;
  }
  g_mutex_unlock(&saver->encoding_mutex);
  g_mutex_clear(&saver->encoding_mutex);
  g_cond_clear(&saver->encoding_cond);

  // Destroy hash tables (matching provided file)
  if (saver->stream_saved_objects) {
    g_hash_table_destroy(saver->stream_saved_objects);
    saver->stream_saved_objects = NULL;
  }
  if (saver->active_recordings) {
    g_hash_table_destroy(saver->active_recordings);
    saver->active_recordings = NULL;
  }

  g_free (saver->output_path);
  saver->output_path = NULL;
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static gboolean
gst_nvdsimagesaver_set_caps (GstBaseTransform * btrans, GstCaps * incaps, GstCaps * outcaps)
{
  GstNvDsImageSaver *saver = GST_NVDSIMAGESAVER (btrans);
  GST_DEBUG_OBJECT (saver, "Caps set");
  return TRUE;
}

static gboolean
gst_nvdsimagesaver_start (GstBaseTransform * btrans)
{
  GstNvDsImageSaver *saver = GST_NVDSIMAGESAVER (btrans);
  GST_DEBUG_OBJECT (saver, "Starting");

  if (!saver->enable) {
      GST_INFO_OBJECT(saver, "Plugin is disabled.");
      return TRUE;
  }

  if (!ensure_output_directory(saver->output_path)) {
      GST_ELEMENT_ERROR (saver, RESOURCE, SETTINGS,
          ("Failed to create output directory: %s", saver->output_path), (nullptr));
      return FALSE;
  }

  // Start encoding thread (matching provided file)
  saver->encoding_thread_running = TRUE;
  saver->encoding_thread = g_thread_new("video-encoder", encoding_thread_func, saver);
  saver->frame_counter = 0;
  saver->next_session_id = 1;

  // Clear state (matching provided file)
  if (saver->stream_saved_objects) g_hash_table_remove_all(saver->stream_saved_objects);
  if (saver->active_recordings) g_hash_table_remove_all(saver->active_recordings);
  g_mutex_lock(&saver->buffer_mutex);
  if (saver->stream_buffers) g_hash_table_remove_all(saver->stream_buffers);
  g_mutex_unlock(&saver->buffer_mutex);

  GST_INFO_OBJECT(saver, "Started with video recording: %s",
                  saver->enable_video_recording ? "enabled" : "disabled");
  return TRUE;
}

static gboolean
gst_nvdsimagesaver_stop (GstBaseTransform * btrans)
{
  GstNvDsImageSaver *saver = GST_NVDSIMAGESAVER (btrans);
  GST_DEBUG_OBJECT (saver, "Stopping");

  // Signal encoding thread to stop (matching provided file)
  g_mutex_lock(&saver->encoding_mutex);
  saver->encoding_thread_running = FALSE;
  g_cond_signal(&saver->encoding_cond);
  g_mutex_unlock(&saver->encoding_mutex);

  // Wait for encoding thread to finish
  if (saver->encoding_thread) {
    g_thread_join(saver->encoding_thread);
    saver->encoding_thread = NULL;
  }

  // Clear state (matching provided file)
  if (saver->stream_saved_objects) g_hash_table_remove_all(saver->stream_saved_objects);
  if (saver->active_recordings) g_hash_table_remove_all(saver->active_recordings);
  g_mutex_lock(&saver->buffer_mutex);
  if (saver->stream_buffers) g_hash_table_remove_all(saver->stream_buffers);
  g_mutex_unlock(&saver->buffer_mutex);

  return TRUE;
}


/**
 * @brief Draws bounding boxes on the NvBufSurface using NvOSD for objects in the frame meta.
 * FIXED VERSION with proper error handling and debugging
 */
static gboolean
draw_bounding_boxes_on_surface(NvBufSurface *surface, NvDsFrameMeta *frame_meta, guint batch_id)
{
    if (!surface || !frame_meta || batch_id >= surface->numFilled) {
        GST_WARNING("Invalid input for drawing bounding boxes: surface=%p, frame_meta=%p, batch_id=%u, numFilled=%u",
                   surface, frame_meta, batch_id, surface ? surface->numFilled : 0);
        return FALSE;
    }
    // Check if surface is in CUDA device memory, required for nvll_osd.
    if (surface->memType != NVBUF_MEM_CUDA_DEVICE) {
         GST_WARNING("Surface memory type (%d) is not CUDA device. NvOSD requires CUDA device memory.", surface->memType);
         return FALSE;
    }
    NvDsMetaList *l_obj = NULL;
    gboolean drawn_any = FALSE;
    // Count objects to draw
    guint num_rects = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
        NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
        if (obj_meta && obj_meta->rect_params.width > 0 && obj_meta->rect_params.height > 0) {
            num_rects++;
            GST_DEBUG("Object %lu: left=%.2f, top=%.2f, width=%.2f, height=%.2f",
                     obj_meta->object_id, obj_meta->rect_params.left, obj_meta->rect_params.top,
                     obj_meta->rect_params.width, obj_meta->rect_params.height);
        }
    }
    if (num_rects == 0) {
        GST_DEBUG("No valid objects to draw for batch_id %u", batch_id);
        return TRUE; // Nothing to draw is not an error
    }
    GST_INFO("Drawing %u bounding boxes on surface (batch_id %u)", num_rects, batch_id);
    // Use NvOSD API
    NvOSDCtxHandle nvosd_context = NULL;
    NvOSD_FrameRectParams frame_rect_params = {0}; // Zero-initialize
    // Allocate array for rectangle parameters
    std::vector<NvOSD_RectParams> rect_params_vector(num_rects);
    NvOSD_RectParams *rect_params = rect_params_vector.data();
    if (!rect_params) {
        GST_ERROR("Failed to allocate memory for NvOSD RectParams.");
        return FALSE;
    }
    // Initialize the rect_params array with more visible settings
    for (guint i = 0; i < num_rects; i++) {
        memset(&rect_params[i], 0, sizeof(NvOSD_RectParams));
        // Make bounding boxes more visible
        rect_params[i].border_width = 4; // Thicker border
        rect_params[i].border_color.red = 1.0;   // Bright red
        rect_params[i].border_color.green = 0.0;
        rect_params[i].border_color.blue = 0.0;
        rect_params[i].border_color.alpha = 1.0; // Fully opaque
        // Optional: Add semi-transparent background
        rect_params[i].bg_color.red = 1.0;
        rect_params[i].bg_color.green = 0.0;
        rect_params[i].bg_color.blue = 0.0;
        rect_params[i].bg_color.alpha = 0.3; // Semi-transparent red background
        rect_params[i].has_bg_color = 1; // Enable background fill
        rect_params[i].reserved = 0;
    }
    // Get surface parameters (move this up before the loop)
    NvBufSurfaceParams *surf_params = &surface->surfaceList[batch_id];
    if (!surf_params || !surf_params->dataPtr) {
        GST_ERROR("Invalid surface parameters or data pointer for batch_id %u.", batch_id);
        // nvll_osd_destroy_context(nvosd_context); // Context not created yet
        return FALSE;
    }
    GST_DEBUG_OBJECT(NULL, "BEFORE DRAWING - Surface dataPtr for batch %u: %p", batch_id, surf_params->dataPtr); // Log dataPtr

    // Populate rectangle parameters from object metadata
    guint rect_idx = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL && rect_idx < num_rects; l_obj = l_obj->next) {
        NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
        if (obj_meta && obj_meta->rect_params.width > 0 && obj_meta->rect_params.height > 0) {
            // Proper casting from gdouble to avoid corruption
            gdouble left = obj_meta->rect_params.left;
            gdouble top = obj_meta->rect_params.top;
            gdouble width = obj_meta->rect_params.width;
            gdouble height = obj_meta->rect_params.height;
            // Ensure coordinates are within bounds and convert to int
            rect_params[rect_idx].left = (int)MAX(0, MIN(left, (gdouble)(surf_params->width - 1)));
            rect_params[rect_idx].top = (int)MAX(0, MIN(top, (gdouble)(surf_params->height - 1)));
            rect_params[rect_idx].width = (int)MAX(1, MIN(width, (gdouble)(surf_params->width - rect_params[rect_idx].left)));
            rect_params[rect_idx].height = (int)MAX(1, MIN(height, (gdouble)(surf_params->height - rect_params[rect_idx].top)));
            GST_DEBUG("Object %lu: orig=(%.2f,%.2f,%.2f,%.2f) -> converted=(%d,%d,%d,%d)",
                    obj_meta->object_id, left, top, width, height,
                    (int)rect_params[rect_idx].left, (int)rect_params[rect_idx].top,
                    (int)rect_params[rect_idx].width, (int)rect_params[rect_idx].height);
            rect_idx++;
            drawn_any = TRUE;
        }
    }
    // Create NvOSD context
    nvosd_context = nvll_osd_create_context();
    if (!nvosd_context) {
        GST_ERROR("Failed to create NvOSD context.");
        return FALSE;
    }
    // Get surface parameters (already done above, kept for clarity)
    if (!surf_params || !surf_params->dataPtr) {
        GST_ERROR("Invalid surface parameters or data pointer for batch_id %u.", batch_id);
        nvll_osd_destroy_context(nvosd_context);
        return FALSE;
    }
    GST_DEBUG("Surface info: width=%u, height=%u, pitch=%u, format=%d, dataPtr=%p",
             surf_params->width, surf_params->height, surf_params->pitch,
             surf_params->colorFormat, surf_params->dataPtr);
    // Fill the FrameRectParams structure
    frame_rect_params.buf_ptr = surf_params;
    frame_rect_params.mode = MODE_GPU;
    frame_rect_params.num_rects = num_rects;
    frame_rect_params.rect_params_list = rect_params;
    // CRITICAL: Synchronize before drawing
    cudaError_t cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        GST_WARNING("cudaDeviceSynchronize before OSD draw failed: %s", cudaGetErrorString(cuda_err));
    } else {
        GST_DEBUG_OBJECT(NULL, "cudaDeviceSynchronize before OSD draw succeeded.");
    }
    // Draw rectangles using NvOSD
    int osd_result = nvll_osd_draw_rectangles(nvosd_context, &frame_rect_params);
    if (osd_result != 0) {
        GST_ERROR("nvll_osd_draw_rectangles failed for batch_id %u with error: %d", batch_id, osd_result);
        nvll_osd_destroy_context(nvosd_context);
        return FALSE;
    } else {
        GST_INFO("✅ Successfully drew %u bounding boxes on batch_id %u using NvOSD.", num_rects, batch_id);
    }
    // CRITICAL: Synchronize after drawing to ensure completion
    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        GST_ERROR("cudaDeviceSynchronize after OSD draw failed: %s", cudaGetErrorString(cuda_err));
        nvll_osd_destroy_context(nvosd_context);
        return FALSE;
    } else {
        GST_DEBUG_OBJECT(NULL, "cudaDeviceSynchronize after OSD draw succeeded. Data should be written to dataPtr %p", surf_params->dataPtr);
    }
    GST_DEBUG_OBJECT(NULL, "AFTER DRAWING AND SYNC - Surface dataPtr for batch %u: %p", batch_id, surf_params->dataPtr); // Log dataPtr again

    // Cleanup
    nvll_osd_destroy_context(nvosd_context);
    return drawn_any;
}

static GstFlowReturn
gst_nvdsimagesaver_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
    GstNvDsImageSaver *saver = GST_NVDSIMAGESAVER (btrans);
    if (!saver->enable) {
        return GST_FLOW_OK;
    }

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
    if (!batch_meta) {
        return GST_FLOW_OK;
    }

    GstMapInfo in_map_info;
    if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
        GST_ERROR_OBJECT (saver, "Failed to map GstBuffer");
        return GST_FLOW_ERROR;
    }

    NvBufSurface *surface = (NvBufSurface *) in_map_info.data;
    if (!surface) {
        gst_buffer_unmap(inbuf, &in_map_info);
        return GST_FLOW_ERROR;
    }

    GstClockTime timestamp = GST_BUFFER_PTS(inbuf);
    nvds_acquire_meta_lock(batch_meta);

    NvDsMetaList *l_frame = NULL;
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        gint stream_id = frame_meta->pad_index; // --- KEY: Get stream ID ---
        guint batch_id = frame_meta->batch_id;
        
        // --- CRITICAL: DRAW BOUNDING BOXES ON THE SURFACE *BEFORE* COPYING FOR RECORDING/SAVING ---
        // This modifies the data in the `surface` buffer directly.
        gboolean drawing_attempted = FALSE;
        if (surface && frame_meta) { // Check if drawing is possible
             // Optional: Add a property to control if boxes are drawn for saving/recording
             // e.g., if (saver->draw_boxes_on_save) {
             drawing_attempted = TRUE;
             if (!draw_bounding_boxes_on_surface(surface, frame_meta, batch_id)) {
                 GST_DEBUG_OBJECT(saver, "Drawing bounding boxes requested but failed or skipped for stream %d, batch %u.", stream_id, batch_id);
                 // Continue processing even if drawing fails, as it's not always critical for saving itself
             } else {
                 GST_DEBUG_OBJECT(saver, "Bounding boxes drawn on surface for stream %d, batch %u.", stream_id, batch_id);
             }
             // Optional: Add a property to control if boxes are drawn for saving/recording
             // }
        } else {
             GST_DEBUG_OBJECT(saver, "Skipping drawing for stream %d, batch %u (surface or frame_meta null).", stream_id, batch_id);
        }
        // --- CHANGED: Buffer current frame for potential video recording (per stream) ---
        if (saver->enable_video_recording) {
            BufferedFrame *buffered_frame = create_buffered_frame(surface, batch_id, saver->frame_counter, timestamp);
            if (buffered_frame) {
                add_frame_to_stream_buffer(saver, stream_id, buffered_frame); // Pass stream_id
            } else {
                GST_WARNING_OBJECT(saver, "Skipping frame %u (batch %u, stream %d) for recording due to format/creation failure.",
                                   saver->frame_counter, batch_id, stream_id);
            }
        }
        // --- END CHANGED ---

        // Check for zone entry events
        gboolean has_zone_entry = FALSE;
        gchar *entering_object_ids = NULL;
        NvDsMetaList *l_user = NULL;
        for (l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *) (l_user->data);
            if (user_meta->base_meta.meta_type == NVDS_EVENT_MSG_META) {
                NvDsEventMsgMeta *event_meta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
                if (is_zone_entry_event(event_meta)) {
                    has_zone_entry = TRUE;
                    if (!entering_object_ids) {
                        entering_object_ids = extract_object_ids_from_events(frame_meta);
                    }
                    break;
                }
            }
        }

        // --- CHANGED: Start video recording if zone entry detected (pass stream_id) ---
        if (has_zone_entry && entering_object_ids && saver->enable_video_recording) {
            GST_DEBUG_OBJECT(saver, "🎬 Start recording for stream %d, objects: %s", stream_id, entering_object_ids);
            start_recording_for_objects(saver, stream_id, entering_object_ids); // Pass stream_id
        }
        // --- END CHANGED ---
        gchar* video_filepath = NULL;
        video_filepath = g_strdup_printf("%svideo_stream_%d_frame_%u_objects_%s.mp4",
                                               saver->output_path, stream_id, saver->frame_counter, entering_object_ids);
        // Process existing image saving logic only if there are events
        gboolean has_events = FALSE;
        for (l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *) (l_user->data);
            if (user_meta->base_meta.meta_type == NVDS_EVENT_MSG_META) {
                has_events = TRUE;
                break;
            }
        }
        if (!has_events) {
            g_free(entering_object_ids);
            continue;
        }

        // --- Image Saving Logic (mostly unchanged from provided file) ---
        gchar *full_frame_path = NULL;
        gboolean full_frame_saved = FALSE;
        if (saver->save_full_frame) {
            gchar full_frame_filename[1024];
            g_snprintf(full_frame_filename, sizeof(full_frame_filename),
                       "%sstream_%u_frame_%u_batch_%u_full.jpg",
                       saver->output_path, stream_id, saver->frame_counter, batch_id);
            if (save_nvmm_buffer_as_jpeg_crop(surface, batch_id, full_frame_filename, 0, 0, 0, 0)) {
                full_frame_path = g_strdup(full_frame_filename);
                full_frame_saved = TRUE;
                GST_INFO_OBJECT(saver, "✅ Full frame saved (stream %d): %s", stream_id, full_frame_filename);
            } else {
                GST_WARNING_OBJECT(saver, "❌ Failed to save full frame (stream %d): %s", stream_id, full_frame_filename);
            }
        }

        NvDsMetaList *l_obj = NULL;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            gchar *crop_path = NULL;
            gboolean crop_saved = FALSE;
            if (saver->save_crops) {
                if (saver->save_each_object_once && is_object_already_saved(saver, obj_meta->object_id, stream_id)) {
                    GST_DEBUG_OBJECT(saver, "Object %" G_GUINT64_FORMAT " already saved for stream %d, skipping.", obj_meta->object_id, stream_id);
                    continue;
                }
                gchar crop_filename[1024];
                g_snprintf(crop_filename, sizeof(crop_filename),
                           "%sstream_%u_frame_%u_batch_%u_obj_%lu_crop.jpg",
                           saver->output_path, stream_id, saver->frame_counter, batch_id, obj_meta->object_id);
                if (save_nvmm_buffer_as_jpeg_crop(surface, batch_id, crop_filename,
                                                  obj_meta->rect_params.left, obj_meta->rect_params.top,
                                                  obj_meta->rect_params.width, obj_meta->rect_params.height)) {
                    crop_path = g_strdup(crop_filename);
                    crop_saved = TRUE;
                    if (saver->save_each_object_once) {
                        mark_object_as_saved(saver, obj_meta->object_id, stream_id);
                        GST_DEBUG_OBJECT(saver, "Marked object %" G_GUINT64_FORMAT " as saved for stream %d", obj_meta->object_id, stream_id);
                    }
                    GST_INFO_OBJECT(saver, "✅ Object crop saved (stream %d): %s", stream_id, crop_filename);
                } else {
                    GST_WARNING_OBJECT(saver, "❌ Failed to save object crop (stream %d): %s", stream_id, crop_filename);
                }
            }

            for (l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
                NvDsUserMeta *user_meta = (NvDsUserMeta *) (l_user->data);
                if (user_meta->base_meta.meta_type == NVDS_EVENT_MSG_META) {
                    NvDsEventMsgMeta *event_meta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
                    if (event_meta) {
                        update_event_meta_with_image_paths(event_meta,
                            full_frame_saved ? full_frame_path : NULL,
                            crop_saved ? crop_path : NULL,
                            video_filepath);
                    }
                }
            }
            g_free(crop_path);
        }
        g_free(full_frame_path);
        g_free(entering_object_ids);
        // --- END Image Saving Logic ---
    }
    saver->frame_counter++; // Increment after processing all frames in the batch
    nvds_release_meta_lock(batch_meta);
    gst_buffer_unmap(inbuf, &in_map_info);
    return GST_FLOW_OK;
}

// --- NEW VIDEO RECORDING FUNCTIONS (Enhanced for streams) ---

static BufferedFrame*
create_buffered_frame(NvBufSurface *surface, guint batch_id, guint frame_number, GstClockTime timestamp)
{
    // ... (Logic from provided file) ...
    if (!surface || batch_id >= surface->numFilled) {
        return NULL;
    }
    NvBufSurfaceParams *params = &surface->surfaceList[batch_id];
    guint width = params->width;
    guint height = params->height;
    BufferedFrame *frame = g_new0(BufferedFrame, 1);
    frame->width = width;
    frame->height = height;
    frame->format = params->colorFormat;
    frame->frame_number = frame_number;
    frame->timestamp = timestamp;

    if (params->colorFormat == NVBUF_COLOR_FORMAT_NV12 || params->colorFormat == NVBUF_COLOR_FORMAT_NV12_ER || params->colorFormat == NVBUF_COLOR_FORMAT_NV12_709_ER || params->colorFormat == NVBUF_COLOR_FORMAT_NV12_709) {
        frame->data_size = width * height * 3 / 2;
    } else if (params->colorFormat == NVBUF_COLOR_FORMAT_RGBA) {
        frame->data_size = width * height * 4;
    } else {
        GST_WARNING("Unsupported format for video recording: %d", params->colorFormat);
        g_free(frame);
        return NULL;
    }
    frame->data = (guint8*)g_malloc(frame->data_size);
    if (!frame->data) {
        g_free(frame);
        return NULL;
    }

    cudaError_t cuda_err;
    if (params->colorFormat == NVBUF_COLOR_FORMAT_NV12 || params->colorFormat == NVBUF_COLOR_FORMAT_NV12_ER || params->colorFormat == NVBUF_COLOR_FORMAT_NV12_709_ER || params->colorFormat == NVBUF_COLOR_FORMAT_NV12_709) {
        cuda_err = cudaMemcpy2D(frame->data, width, params->dataPtr, params->pitch, width, height, cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            GST_ERROR("Failed to copy Y plane: %s", cudaGetErrorString(cuda_err));
            destroy_buffered_frame(frame);
            return NULL;
        }
        guint8 *uv_src = (guint8*)params->dataPtr + (params->pitch * height);
        guint8 *uv_dst = frame->data + (width * height);
        cuda_err = cudaMemcpy2D(uv_dst, width, uv_src, params->pitch, width, height / 2, cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            GST_ERROR("Failed to copy UV plane: %s", cudaGetErrorString(cuda_err));
            destroy_buffered_frame(frame);
            return NULL;
        }
    } else if (params->colorFormat == NVBUF_COLOR_FORMAT_RGBA) {
        cuda_err = cudaMemcpy2D(frame->data, width * 4, params->dataPtr, params->pitch, width * 4, height, cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            GST_ERROR("Failed to copy RGBA data: %s", cudaGetErrorString(cuda_err));
            destroy_buffered_frame(frame);
            return NULL;
        }
    }
    return frame;
}

static void
destroy_buffered_frame(BufferedFrame *frame)
{
    // ... (Logic from provided file) ...
    if (frame) {
        g_free(frame->data);
        g_free(frame);
    }
}

static RecordingSession*
create_recording_session(guint session_id, gint stream_id, guint start_frame, const gchar* object_ids, const gchar* output_path)
{
    // ... (Logic from provided file, plus stream_id) ...
    RecordingSession *session = g_new0(RecordingSession, 1);
    session->session_id = session_id;
    session->stream_id = stream_id; // Store stream ID
    session->start_frame = start_frame;
    session->object_ids = g_string_new(object_ids);
    session->frames = g_queue_new();
    session->frames_recorded = 0;
    session->is_complete = FALSE;
    session->output_filename = g_strdup_printf("%svideo_stream_%d_frame_%u_objects_%s.mp4",
                                               output_path, stream_id, start_frame, object_ids);
    GST_INFO("🎬 Created recording session %u for stream %d: %s", session_id, stream_id, session->output_filename);
    return session;
}

static void
destroy_recording_session(RecordingSession *session)
{
    // ... (Logic from provided file) ...
    if (session) {
        if (session->object_ids) g_string_free(session->object_ids, TRUE);
        if (session->frames) {
            while (!g_queue_is_empty(session->frames)) {
                BufferedFrame *frame = (BufferedFrame*)g_queue_pop_head(session->frames);
                destroy_buffered_frame(frame);
            }
            g_queue_free(session->frames);
        }
        g_free(session->output_filename);
        g_free(session);
    }
}

// --- CHANGED: Stream-aware buffering ---
static GQueue*
get_or_create_stream_buffer(GstNvDsImageSaver *saver, gint stream_id) {
    g_mutex_lock(&saver->buffer_mutex);
    GQueue *buf = (GQueue*)g_hash_table_lookup(saver->stream_buffers, &stream_id);
    if (!buf) {
        buf = g_queue_new();
        gint *key = g_new(gint, 1);
        *key = stream_id;
        g_hash_table_insert(saver->stream_buffers, key, buf);
        GST_DEBUG_OBJECT(saver, "Created new buffer for stream %d", stream_id);
    }
    g_mutex_unlock(&saver->buffer_mutex);
    return buf;
}

static void
add_frame_to_stream_buffer(GstNvDsImageSaver *saver, gint stream_id, BufferedFrame *frame)
{
    GQueue *buf = get_or_create_stream_buffer(saver, stream_id);
    g_mutex_lock(&saver->buffer_mutex);
    g_queue_push_tail(buf, frame);
    while (g_queue_get_length(buf) > saver->max_buffer_frames) {
        BufferedFrame *old_frame = (BufferedFrame*)g_queue_pop_head(buf);
        destroy_buffered_frame(old_frame);
    }
    g_mutex_unlock(&saver->buffer_mutex);
}
// --- END CHANGED ---

static void
start_recording_for_objects(GstNvDsImageSaver *saver, gint stream_id, const gchar* object_ids)
{
    // --- CHANGED: Use stream-specific buffer ---
    guint session_id = saver->next_session_id++;
    RecordingSession *session = create_recording_session(
        session_id, stream_id, saver->frame_counter, object_ids, saver->output_path);

    g_hash_table_insert(saver->active_recordings, GUINT_TO_POINTER(session_id), session);

    // Copy recent frames from the specific stream's buffer
    GQueue *stream_buf = get_or_create_stream_buffer(saver, stream_id);
    g_mutex_lock(&saver->buffer_mutex);
    guint buf_len = g_queue_get_length(stream_buf);
    guint frames_to_copy = MIN(buf_len, VIDEO_RECORD_FRAMES);
    guint skip_frames = buf_len - frames_to_copy;
    GList *buffer_list = g_queue_peek_head_link(stream_buf);
    for (guint i = 0; i < skip_frames && buffer_list; i++) {
        buffer_list = buffer_list->next;
    }

    while (buffer_list && session->frames_recorded < VIDEO_RECORD_FRAMES) {
        BufferedFrame *original = (BufferedFrame*)buffer_list->data;
        BufferedFrame *copy = g_new0(BufferedFrame, 1);
        *copy = *original;
        copy->data = (guint8*)g_malloc(original->data_size);
        if (copy->data) {
            memcpy(copy->data, original->data, original->data_size);
            g_queue_push_tail(session->frames, copy);
            session->frames_recorded++;
        } else {
            g_free(copy); // Free copy struct if data allocation failed
            GST_ERROR_OBJECT(saver, "Failed to allocate memory for frame copy in session %u", session_id);
        }
        buffer_list = buffer_list->next;
    }
    g_mutex_unlock(&saver->buffer_mutex);

    GST_INFO_OBJECT(saver, "🎬 Started video recording session %u for stream %d, objects: %s",
                    session_id, stream_id, object_ids);

    // If we have enough frames immediately, queue for encoding
    if (session->frames_recorded >= VIDEO_RECORD_FRAMES) {
        session->is_complete = TRUE;
        VideoEncodeWorkItem *work_item = g_new0(VideoEncodeWorkItem, 1);
        work_item->session = session;
        work_item->saver = saver;
        g_mutex_lock(&saver->encoding_mutex);
        g_queue_push_tail(saver->encoding_queue, work_item);
        g_cond_signal(&saver->encoding_cond);
        g_mutex_unlock(&saver->encoding_mutex);
    }
    // --- END CHANGED ---
}

static gboolean
is_zone_entry_event(NvDsEventMsgMeta *event_meta)
{
    // ... (Logic from provided file) ...
    if (!event_meta) return FALSE;
    if (event_meta->type == NVDS_EVENT_ENTRY) return TRUE;
    if (event_meta->otherAttrs &&
        (strstr(event_meta->otherAttrs, "entry") || strstr(event_meta->otherAttrs, "zone"))) {
        return TRUE;
    }
    return FALSE;
}

static gchar*
extract_object_ids_from_events(NvDsFrameMeta *frame_meta)
{
    // ... (Logic from provided file) ...
    GString *object_ids = g_string_new("");
    gboolean first = TRUE;
    NvDsMetaList *l_user = NULL;
    for (l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user->data);
        if (user_meta->base_meta.meta_type == NVDS_EVENT_MSG_META) {
            NvDsEventMsgMeta *event_meta = (NvDsEventMsgMeta *)user_meta->user_meta_data;
            if (event_meta) {
                if (!first) g_string_append(object_ids, "_");
                // Use trackingId or objectId as appropriate
                g_string_append_printf(object_ids, "%lu", event_meta->trackingId);
                first = FALSE;
            }
        }
    }
    if (first) {
        g_string_free(object_ids, TRUE);
        return g_strdup("");
    }
    return g_string_free(object_ids, FALSE);
}

static gpointer
encoding_thread_func(gpointer user_data)
{
    // ... (Logic from provided file) ...
    GstNvDsImageSaver *saver = (GstNvDsImageSaver*)user_data;
    GST_INFO_OBJECT(saver, "🎬 Video encoding thread started");
    while (TRUE) {
        g_mutex_lock(&saver->encoding_mutex);
        while (g_queue_is_empty(saver->encoding_queue) && saver->encoding_thread_running) {
            g_cond_wait(&saver->encoding_cond, &saver->encoding_mutex);
        }
        if (!saver->encoding_thread_running) {
            g_mutex_unlock(&saver->encoding_mutex);
            break;
        }
        VideoEncodeWorkItem *work_item = (VideoEncodeWorkItem*)g_queue_pop_head(saver->encoding_queue);
        g_mutex_unlock(&saver->encoding_mutex);

        if (work_item) {
            encode_recording_session(work_item->session, work_item->saver);
            // Remove session from active recordings
            g_hash_table_remove(work_item->saver->active_recordings,
                              GUINT_TO_POINTER(work_item->session->session_id));
            g_free(work_item);
        }
    }
    GST_INFO_OBJECT(saver, "🎬 Video encoding thread stopped");
    return NULL;
}

static void
encode_recording_session(RecordingSession *session, GstNvDsImageSaver *saver)
{
    // ... (Logic from provided file) ...
    GST_INFO_OBJECT(saver, "🎥 Encoding video session %u (stream %d) with %u frames",
                    session->session_id, session->stream_id, session->frames_recorded);

    if (g_queue_is_empty(session->frames)) {
        GST_WARNING_OBJECT(saver, "No frames to encode for session %u", session->session_id);
        return;
    }

    try {
        BufferedFrame *first_frame = (BufferedFrame*)g_queue_peek_head(session->frames);
        if (!first_frame) {
            GST_ERROR_OBJECT(saver, "Failed to get first frame for session %u", session->session_id);
            return;
        }

        cv::VideoWriter writer;
        cv::Size frame_size(first_frame->width, first_frame->height);
        double fps = 25.0; // Adjust as needed or derive from caps
        bool success = writer.open(session->output_filename,
                                   cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                   fps, frame_size);
        if (!success) {
            GST_ERROR_OBJECT(saver, "Failed to create video writer for: %s", session->output_filename);
            return;
        }

        GList *frame_link = g_queue_peek_head_link(session->frames);
        while (frame_link) {
            BufferedFrame *frame = (BufferedFrame*)frame_link->data;
            cv::Mat bgr_frame;
            if (frame->format == NVBUF_COLOR_FORMAT_NV12 || frame->format == NVBUF_COLOR_FORMAT_NV12_ER || frame->format == NVBUF_COLOR_FORMAT_NV12_709_ER || frame->format == NVBUF_COLOR_FORMAT_NV12_709) {
                cv::Mat y_plane(frame->height, frame->width, CV_8UC1, frame->data);
                cv::Mat uv_plane(frame->height / 2, frame->width / 2, CV_8UC2,
                                 frame->data + frame->width * frame->height);
                cv::Mat uv_resized;
                cv::resize(uv_plane, uv_resized, cv::Size(frame->width, frame->height));
                std::vector<cv::Mat> uv_channels;
                cv::split(uv_resized, uv_channels);
                std::vector<cv::Mat> yuv_channels = {y_plane, uv_channels[0], uv_channels[1]};
                cv::Mat yuv_frame;
                cv::merge(yuv_channels, yuv_frame);
                cv::cvtColor(yuv_frame, bgr_frame, cv::COLOR_YUV2BGR);
            } else if (frame->format == NVBUF_COLOR_FORMAT_RGBA) {
                cv::Mat rgba_frame(frame->height, frame->width, CV_8UC4, frame->data);
                cv::cvtColor(rgba_frame, bgr_frame, cv::COLOR_RGBA2BGR);
            } else {
                GST_WARNING_OBJECT(saver, "Unsupported format %d for video encoding in session %u", frame->format, session->session_id);
                frame_link = frame_link->next;
                continue;
            }
            writer.write(bgr_frame);
            frame_link = frame_link->next;
        }
        writer.release();
        GST_INFO_OBJECT(saver, "✅ Video saved successfully: %s", session->output_filename);
    } catch (const cv::Exception& e) {
        GST_ERROR_OBJECT(saver, "OpenCV exception during video encoding for session %u: %s", session->session_id, e.what());
    } catch (...) {
        GST_ERROR_OBJECT(saver, "Unknown exception during video encoding for session %u", session->session_id);
    }
}

static gboolean
save_nvmm_buffer_as_jpeg_crop(NvBufSurface *surface, guint batch_id, const gchar *filename,
                              gdouble crop_left, gdouble crop_top, gdouble crop_width, gdouble crop_height)
{
    if (!surface || batch_id >= surface->numFilled) {
        GST_ERROR("Invalid NvBufSurface or batch_id");
        return FALSE;
    }

    // --- ADDED: Check memory type EARLY ---
    if (surface->memType != NVBUF_MEM_CUDA_DEVICE) {
         GST_ERROR("Expected CUDA device memory, got type %d", surface->memType);
         return FALSE;
    }
    // --- END ADDED ---

    NvBufSurfaceParams *src_params = &surface->surfaceList[batch_id];

    GST_DEBUG("Crop save - Input surface Memory type: %d, Color format: %d (%s)",
              surface->memType, src_params->colorFormat,
              (src_params->colorFormat == NVBUF_COLOR_FORMAT_NV12 || src_params->colorFormat == NVBUF_COLOR_FORMAT_NV12_ER || src_params->colorFormat == NVBUF_COLOR_FORMAT_NV12_709_ER || src_params->colorFormat == NVBUF_COLOR_FORMAT_NV12_709) ? "NV12" :
              (src_params->colorFormat == NVBUF_COLOR_FORMAT_RGBA) ? "RGBA" : "Other/Unknown");

    GST_DEBUG("Crop region: left=%.2f, top=%.2f, width=%.2f, height=%.2f",
              crop_left, crop_top, crop_width, crop_height);

    guint width = src_params->width;
    guint height = src_params->height;

    // Validate dimensions
    if (width == 0 || height == 0) {
        GST_ERROR("Invalid image dimensions: %dx%d", width, height);
        return FALSE;
    }


    // --- CHANGED/ADDED: Handle RGBA format ---
    if (src_params->colorFormat == NVBUF_COLOR_FORMAT_RGBA) {
        guint rgba_size = width * height * 4; // 4 bytes per pixel for RGBA

        // Allocate CPU memory for the RGBA data
        guint8 *cpu_rgba_data = (guint8*)g_malloc(rgba_size);
        if (!cpu_rgba_data) {
            GST_ERROR("Failed to allocate CPU memory for RGBA data");
            return FALSE;
        }

        GST_DEBUG("save_nvmm_buffer_as_jpeg_crop (RGBA): Copying data. src=%p, dst=%p, width_bytes=%u, height=%u, src_pitch=%u, dst_pitch=%u",
                  src_params->dataPtr, cpu_rgba_data, width*4, height, src_params->pitch, width*4);

        // Copy RGBA plane from GPU to CPU
        // RGBA is typically tightly packed, so pitch should be width * 4
        cudaError_t cuda_err = cudaMemcpy2D(
            cpu_rgba_data, width * 4, // dst, dpitch
            src_params->dataPtr, src_params->pitch, // src, spitch
            width * 4, height, // width (in bytes), height
            cudaMemcpyDeviceToHost
        );

        if (cuda_err != cudaSuccess) {
            GST_ERROR("CUDA RGBA plane memcpy failed: %s", cudaGetErrorString(cuda_err));
            g_free(cpu_rgba_data);
            return FALSE;
        }

        gboolean result = FALSE;
        try {
            // Create OpenCV Mat from RGBA data (4 channels)
            cv::Mat rgba_mat(height, width, CV_8UC4, cpu_rgba_data);

            cv::Mat bgr_image;
            // Convert RGBA to BGR (OpenCV discards Alpha by default in this conversion)
            cv::cvtColor(rgba_mat, bgr_image, cv::COLOR_RGBA2BGR);
            // Alternative if you want to use the alpha channel for blending against a background:
            // cv::cvtColor(rgba_mat, bgr_image, cv::COLOR_RGBA2BGRA); // Keeps 4 channels
            // Or simply: cv::cvtColor(rgba_mat, bgr_image, cv::COLOR_BGRA2BGR); // Common alias

            // Crop the image if crop parameters are provided and valid
            if (crop_width > 0 && crop_height > 0) {
                // Ensure crop region is within bounds
                gint x = (gint)MAX(0, crop_left);
                gint y = (gint)MAX(0, crop_top);
                gint w = (gint)MIN(crop_width, (double)(width - x));
                gint h = (gint)MIN(crop_height, (double)(height - y));

                if (w > 0 && h > 0) {
                    cv::Rect crop_rect(x, y, w, h);
                    bgr_image = bgr_image(crop_rect).clone();
                    GST_DEBUG("Cropped RGBA image to region: x=%d, y=%d, w=%d, h=%d", x, y, w, h);
                } else {
                    GST_WARNING("Invalid crop region for RGBA, saving full frame");
                }
            }

            std::vector<int> compression_params;
            compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
            compression_params.push_back(95);

            bool success = cv::imwrite(filename, bgr_image, compression_params);

            if (success) {
                GST_INFO("Successfully saved %s image (from RGBA) to: %s",
                         (crop_width > 0) ? "cropped" : "full", filename);
                result = TRUE;
            } else {
                GST_ERROR("OpenCV imwrite failed for %s (from RGBA)", filename);
            }

        } catch (const cv::Exception& e) {
            GST_ERROR("OpenCV exception during RGBA crop save: %s", e.what());
        } catch (...) {
            GST_ERROR("Unknown exception during RGBA crop save");
        }

        g_free(cpu_rgba_data);
        return result;
    }
    // --- END CHANGED/ADDED: Handle RGBA ---

    // --- Handle NV12 format (existing logic) ---
    if (src_params->colorFormat == NVBUF_COLOR_FORMAT_NV12 ||
        src_params->colorFormat == NVBUF_COLOR_FORMAT_NV12_ER ||
        src_params->colorFormat == NVBUF_COLOR_FORMAT_NV12_709_ER || src_params->colorFormat == NVBUF_COLOR_FORMAT_NV12_709) {

        guint y_size = width * height;
        guint uv_size = width * height / 2;

        // Allocate CPU memory for both Y and UV planes
        guint8 *cpu_y_data = (guint8*)g_malloc(y_size);
        guint8 *cpu_uv_data = (guint8*)g_malloc(uv_size);

        if (!cpu_y_data || !cpu_uv_data) {
            GST_ERROR("Failed to allocate CPU memory for NV12 planes");
            g_free(cpu_y_data);
            g_free(cpu_uv_data);
            return FALSE;
        }
        GST_DEBUG("save_nvmm_buffer_as_jpeg_crop (NV12): Copying Y plane. src=%p, dst=%p, width=%u, height=%u, src_pitch=%u, dst_pitch=%u",
                  src_params->dataPtr, cpu_y_data, width, height, src_params->pitch, width);
        // Copy Y plane from GPU to CPU
        cudaError_t cuda_err = cudaMemcpy2D(
            cpu_y_data, width, src_params->dataPtr, src_params->pitch,
            width, height, cudaMemcpyDeviceToHost
        );

        if (cuda_err != cudaSuccess) {
            GST_ERROR("CUDA Y plane memcpy failed: %s", cudaGetErrorString(cuda_err));
            g_free(cpu_y_data);
            g_free(cpu_uv_data);
            return FALSE;
        }

        // Copy UV plane
        guint8 *uv_src_ptr = (guint8*)src_params->dataPtr + (src_params->pitch * height);

        
        GST_DEBUG("save_nvmm_buffer_as_jpeg_crop (NV12): Copying UV plane. src=%p, dst=%p, width=%u, height=%u, src_pitch=%u, dst_pitch=%u",
                  uv_src_ptr, cpu_uv_data, width, height/2, src_params->pitch, width);
        cuda_err = cudaMemcpy2D(
            cpu_uv_data, width, uv_src_ptr, src_params->pitch,
            width, height / 2, cudaMemcpyDeviceToHost
        );

        if (cuda_err != cudaSuccess) {
            GST_ERROR("CUDA UV plane memcpy failed: %s", cudaGetErrorString(cuda_err));
            g_free(cpu_y_data);
            g_free(cpu_uv_data);
            return FALSE;
        }

        gboolean result = FALSE;
        try {
            // Create full frame OpenCV image (existing NV12 logic)
            cv::Mat y_mat(height, width, CV_8UC1, cpu_y_data);
            cv::Mat uv_mat(height / 2, width / 2, CV_8UC2, cpu_uv_data);

            cv::Mat uv_resized;
            cv::resize(uv_mat, uv_resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

            std::vector<cv::Mat> uv_channels;
            cv::split(uv_resized, uv_channels);

            std::vector<cv::Mat> yuv_channels = {y_mat, uv_channels[0], uv_channels[1]};
            cv::Mat yuv_image;
            cv::merge(yuv_channels, yuv_image);

            cv::Mat bgr_image;
            cv::cvtColor(yuv_image, bgr_image, cv::COLOR_YUV2BGR);

            // Crop the image if crop parameters are provided
            if (crop_width > 0 && crop_height > 0) {
                // Ensure crop region is within bounds (same logic as before)
                gint x = (gint)MAX(0, crop_left);
                gint y = (gint)MAX(0, crop_top);
                gint w = (gint)MIN(crop_width, (double)(width - x));
                gint h = (gint)MIN(crop_height, (double)(height - y));

                if (w > 0 && h > 0) {
                    cv::Rect crop_rect(x, y, w, h);
                    bgr_image = bgr_image(crop_rect).clone();
                    GST_DEBUG("Cropped NV12 image to region: x=%d, y=%d, w=%d, h=%d", x, y, w, h);
                } else {
                    GST_WARNING("Invalid crop region for NV12, saving full frame");
                }
            }

            std::vector<int> compression_params;
            compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
            compression_params.push_back(95);

            bool success = cv::imwrite(filename, bgr_image, compression_params);

            if (success) {
                GST_INFO("Successfully saved %s image (from NV12) to: %s",
                         (crop_width > 0) ? "cropped" : "full", filename);
                result = TRUE;
            } else {
                GST_ERROR("OpenCV imwrite failed for %s (from NV12)", filename);
            }

        } catch (const cv::Exception& e) {
            GST_ERROR("OpenCV exception during NV12 crop save: %s", e.what());
        } catch (...) {
            GST_ERROR("Unknown exception during NV12 crop save");
        }

        g_free(cpu_y_data);
        g_free(cpu_uv_data);
        return result;
    }
    // --- END Handle NV12 format ---

    GST_WARNING("Crop save does not support format %d", src_params->colorFormat);
    return FALSE;
}
// --- END save_nvmm_buffer_as_jpeg_crop ---

/**
 * @brief Check if an object has already been saved for a specific stream
 * --- CHANGED: Added stream_id parameter ---
 */
static gboolean
is_object_already_saved(GstNvDsImageSaver *saver, guint64 object_id, gint stream_id)
{
    if (!saver->stream_saved_objects) return FALSE;

    // Get the hash table for this specific stream
    GHashTable *stream_objects = (GHashTable*)g_hash_table_lookup(saver->stream_saved_objects, &stream_id);

    if (!stream_objects) {
        // No entry for this stream means no objects saved yet for this stream
        return FALSE;
    }

    // Check if the object_id exists in this stream's set
    // Use the address of object_id for g_int64_hash/equal via g_hash_table_contains
    return g_hash_table_contains(stream_objects, &object_id);
}

/**
 * @brief Mark an object as saved for a specific stream
 * --- CHANGED: Added stream_id parameter and manages per-stream sets ---
 */
static void
mark_object_as_saved(GstNvDsImageSaver *saver, guint64 object_id, gint stream_id)
{
    if (!saver->stream_saved_objects) return;

    // Get the hash table for this specific stream
    GHashTable *stream_objects = (GHashTable*)g_hash_table_lookup(saver->stream_saved_objects, &stream_id);

    if (!stream_objects) {
        // If no entry exists for this stream, create a new one
        stream_objects = g_hash_table_new_full(g_int64_hash, g_int64_equal, NULL, NULL); // Keys are guint64 values
        // Create a key for the outer map
        gint *key_stream_id = g_new(gint, 1);
        *key_stream_id = stream_id;
        // Insert the new stream map into the main map
        g_hash_table_insert(saver->stream_saved_objects, key_stream_id, stream_objects);
    }

    // Add the object_id to this stream's set
    // g_hash_table_add expects a pointer to the key value for g_int64_hash/equal
    g_hash_table_add(stream_objects, &object_id);
    GST_DEBUG_OBJECT(saver, "Marked object %" G_GUINT64_FORMAT " as saved for stream %d", object_id, stream_id);
}
// --- END CHANGED ---

/**
 * @brief Updates existing NvDsEventMsgMeta with image paths
 */
static void
update_event_meta_with_image_paths(NvDsEventMsgMeta *event_meta,
                                   const gchar *full_frame_path,
                                   const gchar *crop_path,
                                   const gchar *video_path)
{
    if (!event_meta) return;

    gchar *new_attrs = NULL;

    if (event_meta->otherAttrs) {
        if (full_frame_path && crop_path && video_path) {
            new_attrs = g_strdup_printf("%s;full_frame_path=%s;crop_path=%s;video_path=%s",
                                        event_meta->otherAttrs, full_frame_path, crop_path, video_path);
        } else if (full_frame_path && crop_path) {
            new_attrs = g_strdup_printf("%s;full_frame_path=%s;crop_path=%s",
                                        event_meta->otherAttrs, full_frame_path, crop_path);
        } else if (full_frame_path) {
            new_attrs = g_strdup_printf("%s;full_frame_path=%s",
                                        event_meta->otherAttrs, full_frame_path);
        } else if (crop_path) {
            new_attrs = g_strdup_printf("%s;crop_path=%s",
                                        event_meta->otherAttrs, crop_path);
        }
    } else {
        if (full_frame_path && crop_path) {
            new_attrs = g_strdup_printf("full_frame_path=%s;crop_path=%s",
                                        full_frame_path, crop_path);
        } else if (full_frame_path) {
            new_attrs = g_strdup_printf("full_frame_path=%s", full_frame_path);
        } else if (crop_path) {
            new_attrs = g_strdup_printf("crop_path=%s", crop_path);
        }
    }

    if (new_attrs) {
        g_free(event_meta->otherAttrs);
        event_meta->otherAttrs = new_attrs;
    }
}



// Helper function implementations (mostly unchanged, included for completeness)
static gboolean
ensure_output_directory(const gchar *path)
{
    struct stat st;
    if (stat(path, &st) == 0) {
        if (S_ISDIR(st.st_mode)) {
            return TRUE;
        } else {
            GST_ERROR("Path %s exists but is not a directory", path);
            return FALSE;
        }
    } else {
        if (mkdir(path, 0755) == 0) {
            GST_INFO("Created output directory: %s", path);
            return TRUE;
        } else {
            GST_ERROR("Failed to create output directory: %s", path);
            return FALSE;
        }
    }
}



// ... (save_nvmm_buffer_as_jpeg_crop implementation remains largely the same) ...
// (It uses 

// --- Image Saving Helper Functions (from provided file, unchanged) ---
// ... (Include the full implementations of save_nvmm_buffer_as_jpeg_crop,
// is_object_already_saved, mark_object_as_saved, update_event_meta_with_image_paths,
// int_hash, int_equal, int64_hash, int64_equal, destroy_stream_object_set,
// ensure_output_directory here) ...
// For brevity, I'll mark where they go. You should copy the full function bodies
// from the provided file into this section.

// --- START: Copy implementations from provided file here ---
// static gboolean save_nvmm_buffer_as_jpeg_crop(...) { ... }
// static gboolean ensure_output_directory(...) { ... }
// static gboolean is_object_already_saved(...) { ... }
// static void mark_object_as_saved(...) { ... }
// static void update_event_meta_with_image_paths(...) { ... }
// --- END: Copy implementations from provided file here ---


/* Plugin initialization */
static gboolean
nvdsimagesaver_plugin_init (GstPlugin * plugin)
{
    GST_DEBUG_CATEGORY_INIT (gst_nvdsimagesaver_debug, "nvdsimagesaver", 0, "nvdsimagesaver plugin");
    return gst_element_register (plugin, "nvdsimagesaver", GST_RANK_NONE, GST_TYPE_NVDSIMAGESAVER);
}

/* Plugin definition */
GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsimagesaver,
    DESCRIPTION,
    nvdsimagesaver_plugin_init,
    VERSION,
    LICENSE,
    BINARY_PACKAGE,
    URL
)