/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "gstnvdsimagesaver.h"
#include <sys/stat.h> // For mkdir
#include <unistd.h>   // For access
#include <cstdio>     // For snprintf
#include <opencv2/opencv.hpp> // For image encoding (JPEG)
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include <cuda_runtime.h>
#include <string.h>

GST_DEBUG_CATEGORY_STATIC (gst_nvdsimagesaver_debug);
#define GST_CAT_DEFAULT gst_nvdsimagesaver_debug

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_OUTPUT_PATH,
  PROP_ENABLE,
  PROP_SAVE_FULL_FRAME,
  PROP_SAVE_CROPS,
  PROP_SAVE_EACH_OBJECT_ONCE // This property's behavior is now fixed
};

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 20
#define DEFAULT_OUTPUT_PATH "/tmp/saved_frames/"
#define DEFAULT_ENABLE TRUE
#define DEFAULT_SAVE_FULL_FRAME TRUE
#define DEFAULT_SAVE_CROPS TRUE
#define DEFAULT_SAVE_EACH_OBJECT_ONCE TRUE

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

/* Function prototypes */
static void gst_nvdsimagesaver_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvdsimagesaver_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static GstFlowReturn gst_nvdsimagesaver_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf);
static gboolean gst_nvdsimagesaver_set_caps (GstBaseTransform * btrans, GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_nvdsimagesaver_start (GstBaseTransform * btrans);
static gboolean gst_nvdsimagesaver_stop (GstBaseTransform * btrans);
static void gst_nvdsimagesaver_finalize (GObject * object);

// --- NEW: Helper functions for GHashTable keys/values ---
// Functions to manage gint keys for outer hash table
static guint int_hash(gconstpointer key) {
    return g_int_hash(key);
}
static gboolean int_equal(gconstpointer a, gconstpointer b) {
    return g_int_equal(a, b);
}
// Function to destroy the inner GHashTable when a stream entry is removed
static void destroy_stream_object_set(gpointer data) {
    if (data) {
        g_hash_table_destroy((GHashTable*)data);
    }
}
// --- END NEW ---

// Helper functions for image saving and meta updates
static gboolean save_nvmm_buffer_as_jpeg_crop(NvBufSurface *surface, guint batch_id, const gchar *filename,
                                               gdouble crop_left, gdouble crop_top, gdouble crop_width, gdouble crop_height);
static gboolean ensure_output_directory(const gchar *path);

// --- CHANGED: Updated helper function signatures to include stream_id ---
static gboolean is_object_already_saved(GstNvDsImageSaver *saver, guint64 object_id, gint stream_id);
static void mark_object_as_saved(GstNvDsImageSaver *saver, guint64 object_id, gint stream_id);
// --- END CHANGED ---

static void update_event_meta_with_image_paths(NvDsEventMsgMeta *event_meta,
                                   const gchar *full_frame_path,
                                   const gchar *crop_path);

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
      "Saves images and updates NvDsEventMsgMeta with file paths (only if events exist)",
      "Your Name / NVIDIA");

  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_nvdsimagesaver_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_nvdsimagesaver_get_property);
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_nvdsimagesaver_finalize);

  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id", "Unique ID",
          "Unique ID for the element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_OUTPUT_PATH,
      g_param_spec_string ("output-path", "Output Path",
          "Base directory path to save images (must end with /)",
          DEFAULT_OUTPUT_PATH, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_ENABLE,
      g_param_spec_boolean ("enable", "Enable",
          "Enable image saving",
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
  saver->unique_id = DEFAULT_UNIQUE_ID;
  saver->output_path = g_strdup(DEFAULT_OUTPUT_PATH);
  saver->enable = DEFAULT_ENABLE;
  saver->save_full_frame = DEFAULT_SAVE_FULL_FRAME;
  saver->save_crops = DEFAULT_SAVE_CROPS;
  saver->save_each_object_once = DEFAULT_SAVE_EACH_OBJECT_ONCE;
  saver->frame_counter = 0;

  // --- CHANGED: Initialize the per-stream saved object IDs map ---
  // Outer map: Key is gint* (stream_id), Value is GHashTable* (set of object_ids)
  saver->stream_saved_objects = g_hash_table_new_full(
      int_hash,      // Custom hash for gint*
      int_equal,     // Custom equal for gint*
      g_free,        // Key destroy function (gint* keys are malloced)
      destroy_stream_object_set // Value destroy function (destroys inner hash table)
  );
  // --- END CHANGED ---

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
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

// --- CHANGED: Updated finalize to clean up the new structure ---
static void
gst_nvdsimagesaver_finalize (GObject * object)
{
  GstNvDsImageSaver *saver = GST_NVDSIMAGESAVER (object);
  GST_DEBUG_OBJECT (saver, "Finalizing GstNvDsImageSaver");

  // Destroy the main hash table, which will trigger destroy_stream_object_set for each entry
  if (saver->stream_saved_objects) {
    g_hash_table_destroy(saver->stream_saved_objects);
    saver->stream_saved_objects = NULL;
  }

  g_free (saver->output_path);
  saver->output_path = NULL;
  G_OBJECT_CLASS (parent_class)->finalize (object);
}
// --- END CHANGED ---

static gboolean
gst_nvdsimagesaver_set_caps (GstBaseTransform * btrans, GstCaps * incaps, GstCaps * outcaps)
{
  GstNvDsImageSaver *saver = GST_NVDSIMAGESAVER (btrans);
  GST_DEBUG_OBJECT (saver, "Caps set");
  return TRUE;
}

// --- CHANGED: Updated start to clear the new per-stream structure ---
static gboolean
gst_nvdsimagesaver_start (GstBaseTransform * btrans)
{
  GstNvDsImageSaver *saver = GST_NVDSIMAGESAVER (btrans);
  GST_DEBUG_OBJECT (saver, "Starting");

  if (!saver->enable) {
      GST_INFO_OBJECT(saver, "Image saving is disabled.");
      return TRUE;
  }

  if (!saver->output_path || strlen(saver->output_path) == 0) {
      GST_ELEMENT_ERROR (saver, RESOURCE, SETTINGS,
          ("Output path not provided"), (nullptr));
      return FALSE;
  }

  if (!ensure_output_directory(saver->output_path)) {
      GST_ELEMENT_ERROR (saver, RESOURCE, SETTINGS,
          ("Failed to create or access output directory: %s", saver->output_path), (nullptr));
      return FALSE;
  }

  saver->frame_counter = 0;

  // Clear the saved object IDs per stream at the start
  if (saver->stream_saved_objects) {
    g_hash_table_remove_all(saver->stream_saved_objects);
  }

  GST_INFO_OBJECT(saver, "Image saving enabled. Output path: %s", saver->output_path);
  return TRUE;
}
// --- END CHANGED ---

// --- CHANGED: Updated stop to clear the new per-stream structure ---
static gboolean
gst_nvdsimagesaver_stop (GstBaseTransform * btrans)
{
  GstNvDsImageSaver *saver = GST_NVDSIMAGESAVER (btrans);
  GST_DEBUG_OBJECT (saver, "Stopping");

  // Clear the saved object IDs per stream at the stop
  if (saver->stream_saved_objects) {
    g_hash_table_remove_all(saver->stream_saved_objects);
  }

  return TRUE;
}
// --- END CHANGED ---

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
// --- END CHANGED ---

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
                                   const gchar *crop_path)
{
    if (!event_meta) return;

    gchar *new_attrs = NULL;

    if (event_meta->otherAttrs) {
        if (full_frame_path && crop_path) {
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

static GstFlowReturn
gst_nvdsimagesaver_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
    GstNvDsImageSaver *saver = GST_NVDSIMAGESAVER (btrans);

    if (!saver->enable) {
        GST_DEBUG_OBJECT (saver, "Plugin is disabled, passing through buffer.");
        return GST_FLOW_OK;
    }

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
    if (!batch_meta) {
        GST_DEBUG_OBJECT (saver, "No NvDsBatchMeta found, passing through.");
        return GST_FLOW_OK;
    }

    GstMapInfo in_map_info;
    if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
        GST_ERROR_OBJECT (saver, "Failed to map GstBuffer");
        return GST_FLOW_ERROR;
    }

    NvBufSurface *surface = (NvBufSurface *) in_map_info.data;
    if (!surface) {
        GST_ERROR_OBJECT(saver, "NvBufSurface is NULL");
        gst_buffer_unmap(inbuf, &in_map_info);
        return GST_FLOW_ERROR;
    }

    GstClockTime timestamp = GST_BUFFER_PTS(inbuf);
    if (!GST_CLOCK_TIME_IS_VALID(timestamp)) {
        timestamp = GST_CLOCK_TIME_NONE;
    }

    nvds_acquire_meta_lock(batch_meta);

    NvDsMetaList *l_frame = NULL;
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        gint stream_id = frame_meta->pad_index; // --- KEY CHANGE: Get stream ID ---
        guint batch_id = frame_meta->batch_id;

        // Only process if there are events
        gboolean has_events = FALSE;
        NvDsMetaList *l_user = NULL;

        
        for (l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *) (l_user->data);
            if (user_meta->base_meta.meta_type == NVDS_EVENT_MSG_META) {
                has_events = TRUE;
                break;
            }
        }

        if (!has_events) {
            GST_DEBUG_OBJECT(saver, "No events in frame (stream %d), skipping image saving.", stream_id);
            continue;
        }
        
        

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
                // --- CHANGED: Pass stream_id to the check function ---
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

                    // --- CHANGED: Pass stream_id to the mark function ---
                    if (saver->save_each_object_once) {
                        mark_object_as_saved(saver, obj_meta->object_id, stream_id);
                        GST_DEBUG_OBJECT(saver, "Marked object %" G_GUINT64_FORMAT " as saved for stream %d", obj_meta->object_id, stream_id);
                    }

                    GST_INFO_OBJECT(saver, "✅ Object crop saved (stream %d): %s", stream_id, crop_filename);
                } else {
                    GST_WARNING_OBJECT(saver, "❌ Failed to save object crop (stream %d): %s", stream_id, crop_filename);
                }
            }

            // Update all existing events with image paths for this object/frame
            for (l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
                NvDsUserMeta *user_meta = (NvDsUserMeta *) (l_user->data);
                if (user_meta->base_meta.meta_type == NVDS_EVENT_MSG_META) {
                    NvDsEventMsgMeta *event_meta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
                    if (event_meta) {
                        update_event_meta_with_image_paths(event_meta,
                            full_frame_saved ? full_frame_path : NULL,
                            crop_saved ? crop_path : NULL);
                    }
                }
            }

            g_free(crop_path);
        }

        g_free(full_frame_path);
        saver->frame_counter++;
    }

    nvds_release_meta_lock(batch_meta);
    gst_buffer_unmap(inbuf, &in_map_info);

    return GST_FLOW_OK;
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
// (It uses surface, batch_id, filename, crop coordinates - no changes needed for multi-stream logic here)
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
              (src_params->colorFormat == NVBUF_COLOR_FORMAT_NV12 || src_params->colorFormat == NVBUF_COLOR_FORMAT_NV12_ER || src_params->colorFormat == NVBUF_COLOR_FORMAT_NV12_709_ER) ? "NV12" :
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
        src_params->colorFormat == NVBUF_COLOR_FORMAT_NV12_709_ER) {

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