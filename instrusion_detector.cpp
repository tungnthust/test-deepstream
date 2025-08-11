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
#include <vector>
#include <tuple>

#include <utility>   // Required for std::pair in NvDsImageSaverFrameMeta

// Add for string stream
#include <iostream>
#include <sstream>
#include "nvdsmeta_schema.h"

#define MEMORY_FEATURES "memory:NVMM"
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


// --- Probe Function to Inspect NvDsEventMsgMeta from nvdszonefilter ---
static GstPadProbeReturn
nvdszonefilter_eventmeta_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    if (!batch_meta) {
        g_print("EVENTMETA PROBE: No NvDsBatchMeta found!\n");
        return GST_PAD_PROBE_OK;
    }

    g_print("EVENTMETA PROBE: ===== Checking for NvDsEventMsgMeta =====\n");

    NvDsMetaList *l_frame = NULL;
    guint frame_count = 0;
    guint total_events_found = 0;

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        frame_count++;
        gint stream_id = frame_meta->pad_index;
        guint64 frame_num = frame_meta->frame_num;
        guint64 buf_pts = frame_meta->buf_pts;

        g_print("EVENTMETA PROBE: --- Frame %u (stream_id=%d, frame_num=%lu, pts=%lu) ---\n",
                frame_count - 1, stream_id, frame_num, buf_pts);

        gboolean frame_has_events = FALSE;

        // Check frame-level user metadata for events
        NvDsMetaList *l_user_frame = NULL;
        for (l_user_frame = frame_meta->frame_user_meta_list; l_user_frame != NULL; l_user_frame = l_user_frame->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user_frame->data);

            if (user_meta->base_meta.meta_type == NVDS_EVENT_MSG_META) {
                frame_has_events = TRUE;
                total_events_found++;
                NvDsEventMsgMeta *event_meta = (NvDsEventMsgMeta *)user_meta->user_meta_data;

                if (event_meta) {
                    g_print("EVENTMETA PROBE:   📄 FOUND Frame-Level NvDsEventMsgMeta!\n");
                    g_print("EVENTMETA PROBE:     - Type: %d\n", event_meta->type);
                    g_print("EVENTMETA PROBE:     - Obj Type: %d\n", event_meta->objType);
                    g_print("EVENTMETA PROBE:     - Obj Class ID: %d\n", event_meta->objClassId);
                    g_print("EVENTMETA PROBE:     - Sensor ID: %d\n", event_meta->sensorId);
                    g_print("EVENTMETA PROBE:     - Place ID: %d\n", event_meta->placeId);
                    g_print("EVENTMETA PROBE:     - Module ID: %d\n", event_meta->moduleId);
                    g_print("EVENTMETA PROBE:     - Tracking ID: %lu\n", event_meta->trackingId);
                    g_print("EVENTMETA PROBE:     - Object ID: %s\n", event_meta->objectId ? event_meta->objectId : "(NULL)");
                    g_print("EVENTMETA PROBE:     - Timestamp: %s\n", event_meta->ts ? event_meta->ts : "(NULL)");
                    g_print("EVENTMETA PROBE:     - Frame ID: %d\n", event_meta->frameId);
                    g_print("EVENTMETA PROBE:     - Confidence: %.4f\n", event_meta->confidence);
                    g_print("EVENTMETA PROBE:     - BBox: [%.2f, %.2f, %.2f x %.2f]\n",
                             event_meta->bbox.left, event_meta->bbox.top,
                             event_meta->bbox.width, event_meta->bbox.height);
                    g_print("EVENTMETA PROBE:     - Location: [lat=%.6f, lon=%.6f, alt=%.6f]\n",
                             event_meta->location.lat, event_meta->location.lon, event_meta->location.alt);
                    g_print("EVENTMETA PROBE:     - Coordinate: [x=%.2f, y=%.2f, z=%.2f]\n",
                             event_meta->coordinate.x, event_meta->coordinate.y, event_meta->coordinate.z);
                    g_print("EVENTMETA PROBE:     - Sensor Str: %s\n", event_meta->sensorStr ? event_meta->sensorStr : "(NULL)");
                    g_print("EVENTMETA PROBE:     - Other Attrs: %s\n", event_meta->otherAttrs ? event_meta->otherAttrs : "(NULL)");
                } else {
                    g_print("EVENTMETA PROBE:   ❌ Frame-Level NvDsEventMsgMeta found but data is NULL!\n");
                }
            }
        }

        // Also check object-level metadata for events
        NvDsMetaList *l_obj = NULL;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            
            NvDsMetaList *l_user_obj = NULL;
            for (l_user_obj = obj_meta->obj_user_meta_list; l_user_obj != NULL; l_user_obj = l_user_obj->next) {
                NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user_obj->data);
                
                if (user_meta->base_meta.meta_type == NVDS_EVENT_MSG_META) {
                    frame_has_events = TRUE;
                    total_events_found++;
                    NvDsEventMsgMeta *event_meta = (NvDsEventMsgMeta *)user_meta->user_meta_data;

                    if (event_meta) {
                        g_print("EVENTMETA PROBE:   🎯 FOUND Object-Level NvDsEventMsgMeta!\n");
                        g_print("EVENTMETA PROBE:     - Type: %d\n", event_meta->type);
                        g_print("EVENTMETA PROBE:     - Obj Type: %d\n", event_meta->objType);
                        g_print("EVENTMETA PROBE:     - Obj Class ID: %d\n", event_meta->objClassId);
                        g_print("EVENTMETA PROBE:     - Tracking ID: %lu\n", event_meta->trackingId);
                    }
                }
            }
        }

        if (!frame_has_events) {
             g_print("EVENTMETA PROBE:   🔍 No NvDsEventMsgMeta found for this frame.\n");
        }

        g_print("EVENTMETA PROBE: ------------------------------\n");
    }

    g_print("EVENTMETA PROBE: ===== Summary =====\n");
    g_print("EVENTMETA PROBE: - Frames processed: %u\n", frame_count);
    g_print("EVENTMETA PROBE: - Total NvDsEventMsgMeta found: %u\n", total_events_found);
    g_print("EVENTMETA PROBE: ===================\n\n");

    return GST_PAD_PROBE_OK;
}

int main(int argc, char *argv[]) {
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL,
             *nvdslogger = NULL;
  GstElement *tiler = NULL, *nvvidconv = NULL, *nvosd = NULL, *vidconv = NULL;
  GstElement *encoder = NULL, *muxer = NULL;
  GstElement *postprocessor = NULL;
  GstElement *tracker = NULL;
  GstElement *tee_element = NULL, *msgconv = NULL, *msgbroker = NULL;
  GstElement *queue_video = NULL, *queue_kafka = NULL;
  GstElement *imagesaver = NULL;
  GstElement *nvdskafka = NULL; // <-- Replace msgconv/msgbroker with this

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

  // Load streams from config
  std::unordered_map<guint, StreamConfig> stream_configs;
  if (!load_streams_from_config(config_file_path, stream_configs, num_sources)) {
      g_printerr("Failed to load stream configurations from %s. Exiting.\n", config_file_path);
      return -1;
  }

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

  // Create primary inference
  pgie = gst_element_factory_make("nvinfer", "primary-inference");
  if (!pgie) {
    g_printerr("Unable to create pgie\n");
    return -1;
  }

  // Create tracker
  tracker = gst_element_factory_make("nvtracker", "object-tracker");
  if (!tracker) {
    g_printerr("Unable to create nvtracker\n");
    return -1;
  }
  g_object_set(G_OBJECT(tracker), "tracker-width", 1280, "tracker-height", 1280,
                "user-meta-pool-size", 1024,
               "ll-lib-file", "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so",
               "ll-config-file", "../nvtracker_config.txt",
               "display-tracking-id", TRUE,
               NULL);
  g_print("Set nvtracker config file path to: ../nvtracker_config.txt\n");

  // Create postprocessor (nvdszonefilter)
  postprocessor = gst_element_factory_make("nvdszonefilter", "postprocessor");
  if (!postprocessor) {
    g_printerr("Unable to create postprocessor (nvdszonefilter)\n");
    return -1;
  }
  g_object_set(G_OBJECT(postprocessor), "config-file-path", config_file_path, NULL);
  // g_object_set(G_OBJECT(postprocessor), "draw-zones", TRUE, NULL);
  

  imagesaver = gst_element_factory_make("nvdsimagesaver", "imagesaver");
  if (!imagesaver) {
    g_printerr("Unable to create imagesaver (nvdsimagesaver)\n");
    return -1;
  }

  g_object_set(G_OBJECT(imagesaver), "output-path", "./events_images", NULL);


  // Add probe to postprocessor output
  // GstPad *zonefilter_src_pad = gst_element_get_static_pad(imagesaver, "src");
  // if (zonefilter_src_pad) {
  //     gst_pad_add_probe(zonefilter_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
  //                       nvdszonefilter_eventmeta_probe, NULL, NULL);
  //     g_print("✅ Pad probe added to nvdszonefilter (postprocessor) src pad to inspect NvDsEventMsgMeta.\n");
  //     gst_object_unref(zonefilter_src_pad);
  // }

  // Create tee element for branching
  tee_element = gst_element_factory_make("tee", "tee-branch");
  if (!tee_element) {
    g_printerr("Unable to create tee element\n");
    return -1;
  }

  // Create queue elements
  queue_video = gst_element_factory_make("queue", "queue-video");
  queue_kafka = gst_element_factory_make("queue", "queue-kafka");
  if (!queue_video || !queue_kafka) {
    g_printerr("Unable to create queue elements\n");
    return -1;
  }

  // Create YOUR custom nvdskafka element
  nvdskafka = gst_element_factory_make("nvdskafka", "nvds-kafka-sink");
  if (!nvdskafka) {
    g_printerr("Unable to create nvdskafka element. Is the plugin installed correctly?\n");
    return -1;
  }

  // --- Configure nvdskafka properties ---
  // Set Kafka broker(s) - Use colon ':' for port, comma ',' for multiple brokers
  g_object_set(G_OBJECT(nvdskafka), "brokers", "localhost:8912", NULL);
  // Set Kafka topic
  g_object_set(G_OBJECT(nvdskafka), "topic", "instrusion_detection", NULL); // Keep typo if that's your topic name
  // Optional: Set Client ID

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

  // Create Kafka elements
  msgconv = gst_element_factory_make("nvmsgconv", "nvmsg-converter");
  if (!msgconv) {
    g_printerr("Unable to create nvmsgconv\n");
    return -1;
  }

  msgbroker = gst_element_factory_make("nvmsgbroker", "nvmsg-broker");
  if (!msgbroker) {
    g_printerr("Unable to create nvmsgbroker\n");
    return -1;
  }

  // Note: nvmsgbroker is a sink element, no fakesink needed

  // Configure nvmsgconv
  g_object_set(G_OBJECT(msgconv), 
               "config", "../msgconv_config.txt",  // Use the detailed config
               "payload-type", 0, // JSON
               "metadata-type", 0,
               NULL);

  // Configure nvmsgbroker for Kafka
  g_object_set(G_OBJECT(msgbroker), 
               "proto-lib", "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_kafka_proto.so",
               "conn-str", "localhost;8912",
               "topic", "instrusion_detection",
               "sync", FALSE,
               NULL);

  // Create display elements
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

  // Add source bins
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

  // Configure elements
  g_object_set(G_OBJECT(streammux),
             "batch-size", num_sources,
             "live-source", TRUE,
             "enable-padding", FALSE,
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

  g_object_set(G_OBJECT(pgie), "config-file-path", "../nvinfer_config.txt",
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
  g_print("Kafka messages will be sent to: localhost:8912, topic: instrusion_detection\n");
  g_print("Press Ctrl+C to stop recording and finalize the output file\n");

  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, g_loop);
  gst_object_unref(bus);

  gst_bin_add_many(GST_BIN(pipeline), pgie,
                   tracker, // ADD TRACKER TO BIN
                   nvdslogger, postprocessor, imagesaver, nvdskafka, tiler,
                   nvvidconv, nvosd, vidconv, encoder, muxer, sink, NULL);

  if (!gst_element_link_many(streammux, pgie,
                             tracker, // LINK TRACKER
                             nvdslogger, postprocessor, nvosd, imagesaver, nvdskafka, tiler,
                             nvvidconv, vidconv, encoder, muxer, sink,
                             NULL)) {
    g_printerr("Elements could not be linked. Exiting.\n");
    return -1;
  }


  // // Add all elements to pipeline
  // gst_bin_add_many(GST_BIN(pipeline), pgie, tracker, nvdslogger, postprocessor, imagesaver,
  //                  tee_element, queue_video, queue_kafka,
  //                  msgconv, msgbroker,  // Removed fakesink_kafka
  //                  tiler, nvvidconv, nvosd, vidconv, encoder, muxer, sink, NULL);

  // // Link elements up to tee
  // if (!gst_element_link_many(streammux, pgie, tracker, nvdslogger, postprocessor, imagesaver,
  //                            tee_element, NULL)) {
  //   g_printerr("Elements could not be linked up to tee. Exiting.\n");
  //   return -1;
  // }

  // // Link tee to video path (queue_video -> tiler -> ... -> sink)
  // GstPad *tee_video_pad = gst_element_request_pad_simple(tee_element, "src_%u");
  // GstPad *queue_video_sink_pad = gst_element_get_static_pad(queue_video, "sink");
  // if (gst_pad_link(tee_video_pad, queue_video_sink_pad) != GST_PAD_LINK_OK) {
  //   g_printerr("Failed to link tee to video queue. Exiting.\n");
  //   return -1;
  // }
  // gst_object_unref(tee_video_pad);
  // gst_object_unref(queue_video_sink_pad);

  // if (!gst_element_link_many(queue_video, tiler, nvvidconv, nvosd, vidconv, 
  //                            encoder, muxer, sink, NULL)) {
  //   g_printerr("Video path elements could not be linked. Exiting.\n");
  //   return -1;
  // }

  // // Link tee to Kafka path (queue_kafka -> msgconv -> msgbroker)
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