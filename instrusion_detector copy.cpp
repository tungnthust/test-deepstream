/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier:
 * LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
// main.cpp
#include "gstnvdsmeta.h"
#include <cuda_runtime_api.h>
#include <glib.h>
#include <gst/gst.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <vector>    // <-- ADD THIS
#include <tuple>     // <-- ADD THIS (needed for std::tuple)

#include <utility>   // Required for std::pair in NvDsImageSaverFrameMeta

// Add for string stream
#include <iostream>
#include <sstream>
#include "nvdsmeta_schema.h"

#define MEMORY_FEATURES "memory:NVMM"
#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_BICYCLE 1
#define PGIE_CLASS_ID_PERSON 2
#define PGIE_CLASS_ID_ROADSIGN 3
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 1280
#define MUXER_BATCH_TIMEOUT_USEC 33000
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

static GstElement *g_pipeline = NULL;
static GMainLoop *g_loop = NULL;
static gboolean g_shutdown_requested = FALSE;

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
  GMainLoop *loop = (GMainLoop *)data;
  switch (GST_MESSAGE_TYPE(msg)) {
  case GST_MESSAGE_EOS:
    g_print("End of stream\n");
    g_main_loop_quit(loop);
    break;
  case GST_MESSAGE_WARNING: {
    gchar *debug = NULL;
    GError *error = NULL;
    gst_message_parse_warning(msg, &error, &debug);
    g_printerr("WARNING from element %s: %s\n", GST_OBJECT_NAME(msg->src),
               error->message);
    if (debug)
      g_printerr("Warning details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    break;
  }
  case GST_MESSAGE_ERROR: {
    gchar *debug = NULL;
    GError *error = NULL;
    gst_message_parse_error(msg, &error, &debug);
    g_printerr("ERROR from element %s: %s\n", GST_OBJECT_NAME(msg->src),
               error->message);
    if (debug)
      g_printerr("Error details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    g_main_loop_quit(loop);
    break;
  }
  default:
    break;
  }
  return TRUE;
}

static void signal_handler(int signum) {
  g_print("\nReceived signal %d, initiating graceful shutdown...\n", signum);
  if (g_shutdown_requested) {
    g_print("Shutdown already in progress, forcing exit...\n");
    exit(1);
  }
  g_shutdown_requested = TRUE;
  if (g_pipeline) {
    g_print("Sending EOS to pipeline...\n");
    gst_element_send_event(g_pipeline, gst_event_new_eos());
  }
  g_timeout_add_seconds(10, (GSourceFunc)g_main_loop_quit, g_loop);
}

static void cb_newpad(GstElement *decodebin, GstPad *decoder_src_pad,
                      gpointer data) {
  g_print("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps(decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure(caps, 0);
  const gchar *name = gst_structure_get_name(str);
  GstElement *source_bin = (GstElement *)data;
  GstCapsFeatures *features = gst_caps_get_features(caps, 0);
  g_print("gstname = %s\n", name);
  if (!strncmp(name, "video", 5)) {
    g_print("features = %s\n", gst_caps_features_to_string(features));
    if (gst_caps_features_contains(features, GST_CAPS_FEATURES_NVMM)) {
      GstPad *bin_ghost_pad = gst_element_get_static_pad(source_bin, "src");
      if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad),
                                    decoder_src_pad)) {
        g_printerr("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref(bin_ghost_pad);
    } else {
      g_printerr("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
  gst_caps_unref(caps);
}

static void decodebin_child_added(GstChildProxy *child_proxy, GObject *object,
                                  gchar *name, gpointer user_data) {
  g_print("Decodebin child added: %s\n", name);
  if (g_strrstr(name, "decodebin") == name) {
    g_signal_connect(G_OBJECT(object), "child-added",
                     G_CALLBACK(decodebin_child_added), user_data);
  }
}

static GstElement *create_source_bin(guint index, gchar *uri) {
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = {};
  g_snprintf(bin_name, 15, "source-bin-%02d", index);
  g_print("Creating source bin: %s\n", bin_name);
  bin = gst_bin_new(bin_name);
  uri_decode_bin = gst_element_factory_make("uridecodebin", "uri-decode-bin");
  if (!bin || !uri_decode_bin) {
    g_printerr("One element in source bin could not be created.\n");
    return NULL;
  }
  g_object_set(G_OBJECT(uri_decode_bin), "uri", uri, NULL);
  g_signal_connect(G_OBJECT(uri_decode_bin), "pad-added", G_CALLBACK(cb_newpad),
                   bin);
  g_signal_connect(G_OBJECT(uri_decode_bin), "child-added",
                   G_CALLBACK(decodebin_child_added), bin);
  gst_bin_add(GST_BIN(bin), uri_decode_bin);
  if (!gst_element_add_pad(bin,
                           gst_ghost_pad_new_no_target("src", GST_PAD_SRC))) {
    g_printerr("Failed to add ghost pad in source bin\n");
    return NULL;
  }
  return bin;
}

// --- Add structure and function for config loading ---
struct StreamConfig {
    std::string uri;
};

static gboolean load_streams_from_config(const gchar* config_path, std::unordered_map<guint, StreamConfig>& stream_configs, guint& num_sources) {
    stream_configs.clear();
    num_sources = 0;

    std::ifstream configFile(config_path);
    if (!configFile.is_open()) {
        g_printerr("Failed to open config file: %s\n", config_path);
        return FALSE;
    }

    std::string line;
    guint expected_num_streams = 0;
    gboolean in_global_section = FALSE;
    gboolean in_stream_section = FALSE;
    guint current_stream_id = G_MAXUINT;

    while (std::getline(configFile, line)) {
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);

        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }

        if (line.front() == '[' && line.back() == ']') {
            std::string sectionName = line.substr(1, line.length() - 2);
            in_global_section = (sectionName == "global");
            if (sectionName.rfind("stream", 0) == 0) {
                try {
                    current_stream_id = std::stoul(sectionName.substr(6));
                    in_stream_section = TRUE;
                } catch (const std::exception& e) {
                    g_printerr("Invalid stream section name format: [%s]\n", sectionName.c_str());
                    in_stream_section = FALSE;
                    current_stream_id = G_MAXUINT;
                }
            } else {
                in_stream_section = FALSE;
                current_stream_id = G_MAXUINT;
            }
            continue;
        }

        size_t delimiterPos = line.find('=');
        if (delimiterPos != std::string::npos) {
            std::string key = line.substr(0, delimiterPos);
            std::string value = line.substr(delimiterPos + 1);

            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            if (in_global_section && key == "num_streams") {
                try {
                    expected_num_streams = std::stoul(value);
                } catch (const std::exception& e) {
                    g_printerr("Failed to parse num_streams: %s\n", value.c_str());
                }
            } else if (in_stream_section && key == "stream_url" && current_stream_id != G_MAXUINT) {
                 StreamConfig config;
                 config.uri = value;
                 stream_configs[current_stream_id] = config;
                 g_print("Loaded URI for stream %u: %s\n", current_stream_id, config.uri.c_str());
                 if (current_stream_id >= num_sources) {
                     num_sources = current_stream_id + 1;
                 }
            }
        }
    }

    configFile.close();

    if (stream_configs.size() != expected_num_streams) {
         g_printerr("Warning: Number of loaded streams (%zu) does not match expected num_streams (%u)\n",
                    stream_configs.size(), expected_num_streams);
    }

    if (stream_configs.empty()) {
        g_printerr("No streams were loaded from the config file.\n");
        return FALSE;
    }

    g_print("Successfully loaded %zu stream configurations from %s\n",
            stream_configs.size(), config_path);
    return TRUE;
}
// --- End Add structure and function ---

// --- Define the Meta Type and Structure ---
// MUST MATCH the string and struct used in gstnvdszonefilter.cpp exactly!
// Use extern "C" linkage specification for C function call within C++ file

// --- Simple event message structure for custom metadata ---
typedef struct {
    gchar *event_type;
    guint stream_id;
    guint64 object_id;
    gchar *timestamp;
    gdouble bbox_left;
    gdouble bbox_top;
    gdouble bbox_width;
    gdouble bbox_height;
    gchar *image_path;
    gchar *json_message;
} CustomEventMsgMeta;

// --- Copy function for custom event metadata ---
static gpointer copy_custom_event_meta(gpointer data, gpointer user_data) {
    CustomEventMsgMeta *src = (CustomEventMsgMeta*)data;
    CustomEventMsgMeta *dst = (CustomEventMsgMeta*)g_malloc0(sizeof(CustomEventMsgMeta));
    
    if (!dst) return nullptr;
    
    dst->event_type = src->event_type ? g_strdup(src->event_type) : nullptr;
    dst->stream_id = src->stream_id;
    dst->object_id = src->object_id;
    dst->timestamp = src->timestamp ? g_strdup(src->timestamp) : nullptr;
    dst->bbox_left = src->bbox_left;
    dst->bbox_top = src->bbox_top;
    dst->bbox_width = src->bbox_width;
    dst->bbox_height = src->bbox_height;
    dst->image_path = src->image_path ? g_strdup(src->image_path) : nullptr;
    dst->json_message = src->json_message ? g_strdup(src->json_message) : nullptr;
    
    return dst;
}

// --- Release function for custom event metadata ---
static void release_custom_event_meta(gpointer data, gpointer user_data) {
    CustomEventMsgMeta *meta = (CustomEventMsgMeta*)data;
    if (!meta) return;
    
    if (meta->event_type) g_free(meta->event_type);
    if (meta->timestamp) g_free(meta->timestamp);
    if (meta->image_path) g_free(meta->image_path);
    if (meta->json_message) g_free(meta->json_message);
    
    g_free(meta);
}

// --- Get user meta type for custom event ---
static GQuark get_custom_event_meta_type() {
    static GQuark _custom_event_meta_type = 0;
    if (!_custom_event_meta_type) {
        _custom_event_meta_type = g_quark_from_static_string("CUSTOM_EVENT_MSG_META");
    }
    return _custom_event_meta_type;
}

// --- Helper function to create event message metadata ---
static CustomEventMsgMeta* create_event_msg_meta(const char* event_type, 
                                                  guint stream_id, 
                                                  guint64 object_id,
                                                  gdouble bbox_left, 
                                                  gdouble bbox_top, 
                                                  gdouble bbox_width, 
                                                  gdouble bbox_height,
                                                  const char* image_path = nullptr)
{
    CustomEventMsgMeta *event_meta = (CustomEventMsgMeta*)g_malloc0(sizeof(CustomEventMsgMeta));
    if (!event_meta) {
        return nullptr;
    }
    
    // Basic event information
    event_meta->event_type = g_strdup(event_type);
    event_meta->stream_id = stream_id;
    event_meta->object_id = object_id;
    event_meta->bbox_left = bbox_left;
    event_meta->bbox_top = bbox_top;
    event_meta->bbox_width = bbox_width;
    event_meta->bbox_height = bbox_height;
    
    // Generate timestamp
    struct timeval tv;
    gettimeofday(&tv, NULL);
    event_meta->timestamp = (gchar*)g_malloc0(64);
    snprintf(event_meta->timestamp, 64, "%ld.%06ld", tv.tv_sec, tv.tv_usec);
    
    if (image_path) {
        event_meta->image_path = g_strdup(image_path);
    }
    
    // Create JSON message with all information using string stream
    std::ostringstream json_stream;
    json_stream << "{"
                << "\"event_type\":\"" << event_type << "\","
                << "\"stream_id\":" << stream_id << ","
                << "\"object_id\":" << object_id << ","
                << "\"timestamp\":\"" << event_meta->timestamp << "\","
                << "\"bbox\":{"
                << "\"left\":" << bbox_left << ","
                << "\"top\":" << bbox_top << ","
                << "\"width\":" << bbox_width << ","
                << "\"height\":" << bbox_height
                << "}";
    
    if (image_path) {
        json_stream << ",\"image_path\":\"" << image_path << "\"";
    }
    
    json_stream << "}";
    
    std::string json_string = json_stream.str();
    event_meta->json_message = g_strdup(json_string.c_str());
    
    return event_meta;
}

// --- Custom metadata to event metadata conversion probe ---
static GstPadProbeReturn
custom_to_event_metadata_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    if (!batch_meta) {
        return GST_PAD_PROBE_OK;
    }

    // --- Define metadata types and structures for nvdszonefilter ---
    #define NVDS_USER_FRAME_META_NVDSZONEFILTER (nvds_get_user_meta_type((gchar*)"NVIDIA.DSANALYTICS.NVDSZONEFILTER"))
    #define NVDS_USER_FRAME_META_NVDSIMAGESAVER (nvds_get_user_meta_type((gchar*)"NVIDIA.DSANALYTICS.NVDSIMAGESAVER"))

    typedef struct {
        guint unique_id;
        std::vector<std::tuple<gint, guint64, gdouble, gdouble, gdouble, gdouble>> objects_in_zones;
        // (stream_id, object_id, x, y, width, height)
    } NvDsZoneFilterFrameMeta;

    typedef struct {
        guint unique_id;
        gchar *full_frame_path; // Owned by this struct
        // Pair: (object_id, crop_path). The gchar* path is owned by this struct.
        std::vector<std::pair<guint64, gchar*>> cropped_object_paths;
    } NvDsImageSaverFrameMeta;

    NvDsMetaList *l_frame = NULL;
    guint events_created = 0;

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        
        // Store image paths for correlation
        std::unordered_map<guint64, std::string> object_image_paths;
        std::string full_frame_path;

        // First pass: collect image paths from nvdsimagesaver
        NvDsMetaList *l_user_image = NULL;
        for (l_user_image = frame_meta->frame_user_meta_list; l_user_image != NULL; l_user_image = l_user_image->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user_image->data);
            if (user_meta->base_meta.meta_type == (NvDsMetaType)NVDS_USER_FRAME_META_NVDSIMAGESAVER) {
                NvDsImageSaverFrameMeta *image_meta = (NvDsImageSaverFrameMeta *)user_meta->user_meta_data;
                if (image_meta) {
                    if (image_meta->full_frame_path) {
                        full_frame_path = std::string(image_meta->full_frame_path);
                    }
                    // Store cropped object paths
                    for (size_t i = 0; i < image_meta->cropped_object_paths.size(); i++) {
                        guint64 obj_id = image_meta->cropped_object_paths[i].first;
                        gchar* path = image_meta->cropped_object_paths[i].second;
                        if (path) {
                            object_image_paths[obj_id] = std::string(path);
                        }
                    }
                }
            }
        }

        // Second pass: process zone filter metadata and create event messages
        NvDsMetaList *l_user_zone = NULL;
        for (l_user_zone = frame_meta->frame_user_meta_list; l_user_zone != NULL; l_user_zone = l_user_zone->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user_zone->data);
            
            if (user_meta->base_meta.meta_type == (NvDsMetaType)NVDS_USER_FRAME_META_NVDSZONEFILTER) {
                NvDsZoneFilterFrameMeta *zone_meta = (NvDsZoneFilterFrameMeta *)user_meta->user_meta_data;
                
                if (zone_meta && !zone_meta->objects_in_zones.empty()) {
                    // Create event messages for each object in zone
                    for (size_t i = 0; i < zone_meta->objects_in_zones.size(); i++) {
                        auto& obj_tuple = zone_meta->objects_in_zones[i];
                        guint stream_id = std::get<0>(obj_tuple);
                        guint64 object_id = std::get<1>(obj_tuple);
                        gdouble bbox_left = std::get<2>(obj_tuple);
                        gdouble bbox_top = std::get<3>(obj_tuple);
                        gdouble bbox_width = std::get<4>(obj_tuple);
                        gdouble bbox_height = std::get<5>(obj_tuple);
                        
                        // Find corresponding image path
                        const char* image_path = nullptr;
                        auto it = object_image_paths.find(object_id);
                        if (it != object_image_paths.end()) {
                            image_path = it->second.c_str();
                        } else if (!full_frame_path.empty()) {
                            image_path = full_frame_path.c_str();
                        }
                        
                        // Create event message metadata
                        CustomEventMsgMeta *event_meta = create_event_msg_meta(
                            "zone_entry", stream_id, object_id, 
                            bbox_left, bbox_top, bbox_width, bbox_height, image_path);
                        
                        if (event_meta) {
                            // Create user meta for the event
                            NvDsUserMeta *event_user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
                            if (event_user_meta) {
                                event_user_meta->user_meta_data = (void*)event_meta;
                                event_user_meta->base_meta.meta_type = (NvDsMetaType)get_custom_event_meta_type();
                                event_user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)copy_custom_event_meta;
                                event_user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)release_custom_event_meta;
                                
                                // Add to frame metadata
                                nvds_add_user_meta_to_frame(frame_meta, event_user_meta);
                                events_created++;
                                
                                g_print("KAFKA_PROBE: Created event message for object_id=%lu in stream=%u (bbox=%.2f,%.2f,%.2f,%.2f)\n",
                                        object_id, stream_id, bbox_left, bbox_top, bbox_width, bbox_height);
                                if (image_path) {
                                    g_print("KAFKA_PROBE:   - Image path: %s\n", image_path);
                                }
                                g_print("KAFKA_PROBE:   - JSON: %s\n", event_meta->json_message);
                            } else {
                                // Clean up if we couldn't get user meta
                                release_custom_event_meta(event_meta, nullptr);
                            }
                        }
                    }
                }
            }
        }
    }

    if (events_created > 0) {
        g_print("KAFKA_PROBE: Created %u event messages for Kafka transmission\n", events_created);
    }

    return GST_PAD_PROBE_OK;
}

// --- General metadata probe (for debugging) ---
static GstPadProbeReturn
general_metadata_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    if (!batch_meta) {
        g_print("PROBE: No batch metadata found\n");
        return GST_PAD_PROBE_OK;
    }

    g_print("PROBE: ===== General Metadata Check =====\n");

    // --- Define metadata types and structures for nvdszonefilter ---
    #define NVDS_USER_FRAME_META_NVDSZONEFILTER (nvds_get_user_meta_type((gchar*)"NVIDIA.DSANALYTICS.NVDSZONEFILTER"))

    typedef struct {
        guint unique_id;
        std::vector<std::tuple<gint, guint64, gdouble, gdouble, gdouble, gdouble>> objects_in_zones;
        // (stream_id, object_id, x, y, width, height)
    } NvDsZoneFilterFrameMeta;


    // --- Define metadata types and structures for nvdsimagesaver ---
    #define NVDS_USER_FRAME_META_NVDSIMAGESAVER (nvds_get_user_meta_type((gchar*)"NVIDIA.DSANALYTICS.NVDSIMAGESAVER"))

    typedef struct {
        guint unique_id;
        gchar *full_frame_path; // Owned by this struct
        // Pair: (object_id, crop_path). The gchar* path is owned by this struct.
        std::vector<std::pair<guint64, gchar*>> cropped_object_paths;
    } NvDsImageSaverFrameMeta;


    NvDsMetaList *l_frame = NULL;
    guint frame_count = 0;
    guint total_zone_objects = 0;
    guint frames_with_image_paths = 0; // Counter for frames with image saver meta

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        frame_count++;

        g_print("PROBE: --- Frame %u (stream_id=%u, frame_num=%u) ---\n",
                frame_count - 1, frame_meta->pad_index, frame_meta->frame_num);

        // --- Check for Zone Filter Metadata ---
        gboolean found_zone_meta = FALSE;
        NvDsMetaList *l_user_zone = NULL;
        for (l_user_zone = frame_meta->frame_user_meta_list; l_user_zone != NULL; l_user_zone = l_user_zone->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user_zone->data);

            if (user_meta->base_meta.meta_type == (NvDsMetaType)NVDS_USER_FRAME_META_NVDSZONEFILTER) {
                found_zone_meta = TRUE;
                NvDsZoneFilterFrameMeta *zone_meta = (NvDsZoneFilterFrameMeta *)user_meta->user_meta_data;

                if (zone_meta) {
                    g_print("PROBE:   ✅ FOUND Zone Filter Metadata!\n");
                    g_print("PROBE:     - Plugin Unique ID: %u\n", zone_meta->unique_id);
                    g_print("PROBE:     - Objects in zone: %zu\n", zone_meta->objects_in_zones.size());

                    total_zone_objects += zone_meta->objects_in_zones.size();

                    // Print details of each object in zone
                    for (size_t i = 0; i < zone_meta->objects_in_zones.size(); i++) {
                        auto& obj_tuple = zone_meta->objects_in_zones[i];
                        g_print("PROBE:       Zone Object[%zu]: stream_id=%d, obj_id=%lu, bbox=(%.2f,%.2f,%.2f,%.2f)\n",
                                i,
                                std::get<0>(obj_tuple),  // stream_id
                                std::get<1>(obj_tuple),  // object_id
                                std::get<2>(obj_tuple),  // left
                                std::get<3>(obj_tuple),  // top
                                std::get<4>(obj_tuple),  // width
                                std::get<5>(obj_tuple)); // height
                    }
                } else {
                    g_print("PROBE:   ❌ Zone metadata found but data is NULL!\n");
                }
                // Found the correct meta type for this frame, no need to check further
                break;
            }
        }

        if (!found_zone_meta) {
            g_print("PROBE:   🔍 No zone filter metadata found for this frame.\n");
        }

        // --- Check for Image Saver Metadata ---
        gboolean found_image_meta = FALSE;
        NvDsMetaList *l_user_image = NULL;
        for (l_user_image = frame_meta->frame_user_meta_list; l_user_image != NULL; l_user_image = l_user_image->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user_image->data);

            if (user_meta->base_meta.meta_type == (NvDsMetaType)NVDS_USER_FRAME_META_NVDSIMAGESAVER) {
                found_image_meta = TRUE;
                NvDsImageSaverFrameMeta *image_meta = (NvDsImageSaverFrameMeta *)user_meta->user_meta_data;

                if (image_meta) {
                    frames_with_image_paths++; // Increment counter
                    g_print("PROBE:   🖼️  FOUND Image Saver Metadata!\n");
                    g_print("PROBE:     - Plugin Unique ID: %u\n", image_meta->unique_id);
                    
                    if (image_meta->full_frame_path) {
                         g_print("PROBE:     - Full Frame Path: %s\n", image_meta->full_frame_path);
                    } else {
                         g_print("PROBE:     - Full Frame Path: (NULL)\n");
                    }

                    g_print("PROBE:     - Number of Cropped Objects: %zu\n", image_meta->cropped_object_paths.size());

                    // Print details of each cropped object path
                    for (size_t i = 0; i < image_meta->cropped_object_paths.size(); i++) {
                        guint64 obj_id = image_meta->cropped_object_paths[i].first;
                        gchar* path = image_meta->cropped_object_paths[i].second; // Could be NULL
                        if (path) {
                            g_print("PROBE:       Crop[%zu]: obj_id=%lu, path=%s\n", i, obj_id, path);
                        } else {
                             g_print("PROBE:       Crop[%zu]: obj_id=%lu, path=(NULL)\n", i, obj_id);
                        }
                    }
                } else {
                    g_print("PROBE:   ❌ Image saver metadata found but data is NULL!\n");
                }
                // Found the correct meta type for this frame, no need to check further
                break;
            }
        }

        if (!found_image_meta) {
            g_print("PROBE:   🔍 No image saver metadata found for this frame.\n");
        }

        g_print("PROBE: ------------------------------\n");
    }

    g_print("PROBE: ===== Summary =====\n");
    g_print("PROBE: - Frames processed: %u\n", frame_count);
    g_print("PROBE: - Total objects in zones (from zonefilter): %u\n", total_zone_objects);
    g_print("PROBE: - Frames with image paths (from imagesaver): %u\n", frames_with_image_paths);
    g_print("PROBE: ===================\n\n");

    return GST_PAD_PROBE_OK;
}
// --- End of Probe Function ---

int main(int argc, char *argv[]) {
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL,
             *nvdslogger = NULL;
  GstElement *tiler = NULL, *nvvidconv = NULL, *nvosd = NULL, *vidconv = NULL;
  GstElement *encoder = NULL, *muxer = NULL;
  GstElement *postprocessor = NULL, *imagesaver = NULL;
  // --- ADD TRACKER ELEMENT ---
  GstElement *tracker = NULL;
  // --- ADD KAFKA ELEMENTS ---
  GstElement *tee_element = NULL, *msgconv = NULL, *msgbroker = NULL;
  GstElement *queue_video = NULL, *queue_kafka = NULL;
  // --- END ADD TRACKER ELEMENT ---
  GstBus *bus = NULL;
  guint bus_watch_id;
  guint i;
  guint num_sources = 0;
  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);
  const gchar *new_mux_str = g_getenv("USE_NEW_NVSTREAMMUX");
  gboolean use_new_mux = !g_strcmp0(new_mux_str, "yes");

  if (argc != 2) {
    g_printerr("Usage: %s <path_to_nvdszonefilter_config.ini>\n", argv[0]);
    return -1;
  }
  const gchar *config_file_path = argv[1];

  // --- Load streams from config ---
  std::unordered_map<guint, StreamConfig> stream_configs;
  if (!load_streams_from_config(config_file_path, stream_configs, num_sources)) {
      g_printerr("Failed to load stream configurations from %s. Exiting.\n", config_file_path);
      return -1;
  }
  // --- End Load streams ---

  gst_init(&argc, &argv);
  g_loop = g_main_loop_new(NULL, FALSE);
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  pipeline = gst_pipeline_new("rtsp-detection-pipeline");
  g_pipeline = pipeline;
  streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
  if (!pipeline || !streammux) {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
  }

  pgie = gst_element_factory_make("nvinfer", "primary-inference");
  if (!pgie) {
    g_printerr("Unable to create pgie\n");
    return -1;
  }

  // --- CREATE TRACKER ELEMENT ---
  tracker = gst_element_factory_make("nvtracker", "object-tracker");
  if (!tracker) {
    g_printerr("Unable to create nvtracker\n");
    return -1;
  }
  g_object_set(G_OBJECT(tracker), "tracker-width", 1280, "tracker-height", 1280,
                "user-meta-pool-size", 1024,
               "ll-lib-file", "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so",
               "ll-config-file", "../nvtracker_config.txt", // Use local config
               "display-tracking-id", TRUE,
               NULL);
  g_print("Set nvtracker config file path to: ../nvtracker_config.txt\n");
  // --- END CREATE TRACKER ELEMENT ---

  postprocessor = gst_element_factory_make("nvdszonefilter", "postprocessor");
  if (!postprocessor) {
    g_printerr("Unable to create postprocessor (nvdszonefilter)\n");
    return -1;
  }

  g_object_set(G_OBJECT(postprocessor), "config-file-path", config_file_path, NULL);

  imagesaver = gst_element_factory_make("nvdsimagesaver", "imagesaver");
  if (!imagesaver) {
    g_printerr("Unable to create imagesaver (nvdsimagesaver)\n");
    return -1;
  }

  g_object_set(G_OBJECT(imagesaver), "output-path", "./events_images", NULL);

  // --- CREATE TEE ELEMENT FOR BRANCHING ---
  tee_element = gst_element_factory_make("tee", "tee-branch");
  if (!tee_element) {
    g_printerr("Unable to create tee element\n");
    return -1;
  }

  // --- CREATE QUEUE ELEMENTS ---
  queue_video = gst_element_factory_make("queue", "queue-video");
  queue_kafka = gst_element_factory_make("queue", "queue-kafka");
  if (!queue_video || !queue_kafka) {
    g_printerr("Unable to create queue elements\n");
    return -1;
  }

  // Configure queues
  g_object_set(G_OBJECT(queue_video), 
               "max-size-buffers", 100,
               "max-size-bytes", 0,
               "max-size-time", 0,
               NULL);

  g_object_set(G_OBJECT(queue_kafka), 
               "max-size-buffers", 100,
               "max-size-bytes", 0,
               "max-size-time", 0,
               NULL);

  // --- CREATE KAFKA ELEMENTS ---
  msgconv = gst_element_factory_make("nvmsgconv", "nvmsg-converter");
  if (!msgconv) {
    g_printerr("Unable to create nvmsgconv\n");
    return -1;
  }

  msgbroker = gst_element_factory_make("nvmsgbroker", "nvmsg-broker");
  if (!msgbroker) {
    g_printerr("Unable to nvmsgbroker\n");
    return -1;
  }

  // Configure nvmsgconv
  g_object_set(G_OBJECT(msgconv), 
               "config", "../msgconv_config.txt",
               "payload-type", 0, // JSON
               NULL);

  // Configure nvmsgbroker for Kafka
  g_object_set(G_OBJECT(msgbroker), 
               "proto-lib", "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_kafka_proto.so",
               "conn-str", "localhost;8912", // Kafka broker address
               "topic", "instrusion_detection", // Kafka topic
               "sync", FALSE,
               NULL);

  // --- ADD PROBE TO CONVERT CUSTOM METADATA TO EVENT METADATA ---
  GstPad *imagesaver_src_pad = gst_element_get_static_pad(imagesaver, "src");
  if (!imagesaver_src_pad) {
      g_printerr("Could not get 'src' pad from imagesaver\n");
      return -1;
  } else {
      // Add the conversion probe to create event metadata for Kafka
      gst_pad_add_probe(imagesaver_src_pad, GST_PAD_PROBE_TYPE_BUFFER, 
                        custom_to_event_metadata_probe, NULL, NULL);
      g_print("Added custom-to-event metadata conversion probe to imagesaver src pad.\n");
      
      // Also add general probe for debugging
      // gst_pad_add_probe(imagesaver_src_pad, GST_PAD_PROBE_TYPE_BUFFER, 
      //                   general_metadata_probe, NULL, NULL);
      // g_print("Added general metadata probe to imagesaver src pad.\n");
      
      gst_object_unref(imagesaver_src_pad);
  }

  tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");
  if (!tiler) {
    g_printerr("Unable to create tiler\n");
    return -1;
  }

  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
  if (!nvvidconv) {
    g_printerr("Unable to create nvvidconv\n");
    return -1;
  }

  nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
  if (!nvosd) {
    g_printerr("Unable to create nvosd\n");
    return -1;
  }

  nvdslogger = gst_element_factory_make("nvdslogger", "nvds-logger");
  if (!nvdslogger) {
    g_printerr("Unable to create nvdslogger\n");
    return -1;
  }

  vidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter2");
  if (!vidconv) {
    g_printerr("Unable to create vidconv\n");
    return -1;
  }

  encoder = gst_element_factory_make("x264enc", "x264-encoder");
  if (!encoder) {
    g_printerr("Unable to create x264 encoder\n");
    return -1;
  }

  muxer = gst_element_factory_make("qtmux", "qtmux");
  if (!muxer) {
    g_printerr("Unable to create qtmux\n");
    return -1;
  }

  sink = gst_element_factory_make("filesink", "filesink");
  if (!sink) {
    g_printerr("Unable to create filesink\n");
    return -1;
  }
  gst_bin_add(GST_BIN(pipeline), streammux);

  for (const auto& pair : stream_configs) {
      guint stream_id = pair.first;
      const StreamConfig& config = pair.second;
      GstPad *sinkpad, *srcpad;
      gchar pad_name[16] = {};
      GstElement *source_bin = create_source_bin(stream_id, const_cast<gchar*>(config.uri.c_str()));
      if (!source_bin) {
        g_printerr("Failed to create source bin %d (URI: %s). Exiting.\n", stream_id, config.uri.c_str());
        return -1;
      }
      gst_bin_add(GST_BIN(pipeline), source_bin);
      g_snprintf(pad_name, 15, "sink_%u", stream_id);
      sinkpad = gst_element_request_pad_simple(streammux, pad_name);
      if (!sinkpad) {
        g_printerr("Streammux request sink pad failed for %s. Exiting.\n", pad_name);
        return -1;
      }
      srcpad = gst_element_get_static_pad(source_bin, "src");
      if (!srcpad) {
        g_printerr("Failed to get src pad of source bin %d. Exiting.\n", stream_id);
        return -1;
      }
      if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link source bin %d to stream muxer. Exiting.\n", stream_id);
        return -1;
      }
      gst_object_unref(srcpad);
      gst_object_unref(sinkpad);
  }



  g_object_set(G_OBJECT(streammux),
             "batch-size", num_sources,
             "live-source", TRUE,
             "width", MUXER_OUTPUT_WIDTH,
             "height", MUXER_OUTPUT_HEIGHT,
             "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC,
             "buffer-pool-size", 128,
             "batched-metadata-pool-size", 1024,
             "gpu-id", 0,
             "nvbuf-memory-type", 0,
             NULL);

  guint tiler_rows = (guint)sqrt(num_sources);
  guint tiler_columns = (guint)ceil(1.0 * num_sources / tiler_rows);
  g_object_set(G_OBJECT(tiler), "rows", tiler_rows, "columns", tiler_columns,
               "width", MUXER_OUTPUT_WIDTH, "height", MUXER_OUTPUT_HEIGHT,
               NULL);

  g_object_set(G_OBJECT(pgie), "config-file-path", "../nvinfer_config.txt", // Adjust path
               NULL);

  g_object_set(G_OBJECT(nvdslogger), "fps-measurement-interval-sec", 2, NULL);

  g_object_set(G_OBJECT(encoder), "bitrate", 4000, NULL);

  g_object_set(G_OBJECT(sink), "location", "output.mp4", NULL);
  g_object_set(G_OBJECT(sink), "sync", FALSE, NULL);
  
  g_print("Playing streams (loaded from %s):\n", config_file_path);
  for (const auto& pair : stream_configs) {
      g_print("  Stream ID %u: %s\n", pair.first, pair.second.uri.c_str());
  }
  g_print("Output will be saved to: output.mp4\n");
  g_print("Kafka messages will be sent to topic: deepstream-events\n");
  g_print("Press Ctrl+C to stop recording and finalize the output file\n");

  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, g_loop);
  gst_object_unref(bus);

  // --- ADD ALL ELEMENTS TO PIPELINE ---
  gst_bin_add_many(GST_BIN(pipeline), pgie,
                   tracker, // ADD TRACKER TO BIN
                   nvdslogger, postprocessor, imagesaver, 
                   tee_element, queue_video, queue_kafka, // ADD TEE AND QUEUES
                   msgconv, msgbroker, // ADD KAFKA ELEMENTS
                   tiler, nvvidconv, nvosd, vidconv, encoder, muxer, sink, NULL);

  // --- LINK ELEMENTS UP TO TEE ---
  if (!gst_element_link_many(streammux, pgie,
                             tracker, // LINK TRACKER
                             nvdslogger, postprocessor, imagesaver,
                             tee_element, // LINK TO TEE
                             NULL)) {
    g_printerr("Elements could not be linked up to tee. Exiting.\n");
    return -1;
  }

  // --- LINK TEE TO VIDEO PATH (queue_video -> tiler -> ... -> sink) ---
  GstPad *tee_video_pad = gst_element_request_pad_simple(tee_element, "src_%u");
  GstPad *queue_video_sink_pad = gst_element_get_static_pad(queue_video, "sink");
  if (gst_pad_link(tee_video_pad, queue_video_sink_pad) != GST_PAD_LINK_OK) {
    g_printerr("Failed to link tee to video queue. Exiting.\n");
    return -1;
  }
  gst_object_unref(tee_video_pad);
  gst_object_unref(queue_video_sink_pad);

  if (!gst_element_link_many(queue_video, tiler, nvvidconv, nvosd, vidconv, 
                             encoder, muxer, sink, NULL)) {
    g_printerr("Video path elements could not be linked. Exiting.\n");
    return -1;
  }

  // --- LINK TEE TO KAFKA PATH (queue_kafka -> msgconv -> msgbroker) ---
  // GstPad *tee_kafka_pad = gst_element_request_pad_simple(tee_element, "src_%u");
  // GstPad *queue_kafka_sink_pad = gst_element_get_static_pad(queue_kafka, "sink");
  // if (gst_pad_link(tee_kafka_pad, queue_kafka_sink_pad) != GST_PAD_LINK_OK) {
  //   g_printerr("Failed to link tee to kafka queue. Exiting.\n");
  //   return -1;
  // }
  // gst_object_unref(tee_kafka_pad);
  // gst_object_unref(queue_kafka_sink_pad);

  // if (!gst_element_link_many(queue_kafka, msgconv, msgbroker, NULL)) {
  //   g_printerr("Kafka path elements could not be linked. Exiting.\n");
  //   return -1;
  // }

  g_print("Starting pipeline\n");
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  g_print("Running...\n");
  g_main_loop_run(g_loop);

  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(g_loop);
  gst_deinit();
  g_print("Program finished. Output file 'output.mp4' should be properly finalized.\n");
  return 0;
}