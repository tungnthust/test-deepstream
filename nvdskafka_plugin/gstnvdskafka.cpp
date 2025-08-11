/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstnvdskafka.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <string.h>
#include <uuid/uuid.h>
#include <sys/time.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <functional> // Added for std::function

GST_DEBUG_CATEGORY_STATIC(gst_nvds_kafka_debug);
#define GST_CAT_DEFAULT gst_nvds_kafka_debug

/* Properties enum */
enum {
  PROP_0,
  PROP_BROKERS,
  PROP_TOPIC,
  PROP_CLIENT_ID,
  PROP_CONFIG_FILE,
  PROP_CUSTOM_TEMPLATE,
  PROP_MSG_FORMAT,
  PROP_ENABLE_DEBUG,
  PROP_ASYNC_SEND,
  PROP_BATCH_SIZE,
  PROP_FLUSH_TIMEOUT,
  PROP_INCLUDE_ALL_ATTRS,
  PROP_PARSE_OTHER_ATTRS,
  PROP_OTHER_ATTRS_SEPARATOR,
  PROP_KEY_VALUE_SEPARATOR
};

/* Default values */
#define DEFAULT_BROKERS "localhost:9092"
#define DEFAULT_TOPIC "deepstream-events"
#define DEFAULT_CLIENT_ID "nvds-kafka-plugin"
#define DEFAULT_MSG_FORMAT GST_NVDS_KAFKA_MSG_FORMAT_FULL
#define DEFAULT_ASYNC_SEND TRUE
#define DEFAULT_BATCH_SIZE 100
#define DEFAULT_FLUSH_TIMEOUT 5000
#define DEFAULT_INCLUDE_ALL_ATTRS TRUE
#define DEFAULT_PARSE_OTHER_ATTRS TRUE
#define DEFAULT_OTHER_ATTRS_SEPARATOR ";"
#define DEFAULT_KEY_VALUE_SEPARATOR "="

/* Message format GEnum */
#define GST_TYPE_NVDS_KAFKA_MSG_FORMAT (gst_nvds_kafka_msg_format_get_type())
static GType
gst_nvds_kafka_msg_format_get_type(void)
{
  static GType msg_format_type = 0;
  if (!msg_format_type) {
    static const GEnumValue msg_format_types[] = {
      {GST_NVDS_KAFKA_MSG_FORMAT_MINIMAL, "Minimal JSON format", "minimal"},
      {GST_NVDS_KAFKA_MSG_FORMAT_FULL, "Full JSON format with all fields", "full"},
      {GST_NVDS_KAFKA_MSG_FORMAT_CUSTOM, "Custom JSON format from template", "custom"},
      {0, NULL, NULL}
    };
    msg_format_type = g_enum_register_static("GstNvdsKafkaMsgFormat", msg_format_types);
  }
  return msg_format_type;
}

/* Pad templates */
static GstStaticPadTemplate gst_nvds_kafka_sink_template =
    GST_STATIC_PAD_TEMPLATE("sink",
        GST_PAD_SINK,
        GST_PAD_ALWAYS,
        GST_STATIC_CAPS("video/x-raw(memory:NVMM), "
            "width=(int)[1,MAX], height=(int)[1,MAX], "
            "framerate=(fraction)[0/1,MAX]; "
            "video/x-raw, "
            "width=(int)[1,MAX], height=(int)[1,MAX], "
            "framerate=(fraction)[0/1,MAX]"));

static GstStaticPadTemplate gst_nvds_kafka_src_template =
    GST_STATIC_PAD_TEMPLATE("src",
        GST_PAD_SRC,
        GST_PAD_ALWAYS,
        GST_STATIC_CAPS("video/x-raw(memory:NVMM), "
            "width=(int)[1,MAX], height=(int)[1,MAX], "
            "framerate=(fraction)[0/1,MAX]; "
            "video/x-raw, "
            "width=(int)[1,MAX], height=(int)[1,MAX], "
            "framerate=(fraction)[0/1,MAX]"));

/* Class initialization */
static void gst_nvds_kafka_class_init(GstNvdsKafkaClass *klass);
static void gst_nvds_kafka_init(GstNvdsKafka *nvds_kafka);
static void gst_nvds_kafka_dispose(GObject *object);
static void gst_nvds_kafka_finalize(GObject *object);
static void gst_nvds_kafka_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec);
static void gst_nvds_kafka_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec);

/* GstBaseTransform methods */
static gboolean gst_nvds_kafka_start(GstBaseTransform *trans);
static gboolean gst_nvds_kafka_stop(GstBaseTransform *trans);
static GstFlowReturn gst_nvds_kafka_transform_ip(GstBaseTransform *trans, GstBuffer *buf);

#define gst_nvds_kafka_parent_class parent_class
G_DEFINE_TYPE(GstNvdsKafka, gst_nvds_kafka, GST_TYPE_BASE_TRANSFORM);

static void
gst_nvds_kafka_class_init(GstNvdsKafkaClass *klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);
  GstBaseTransformClass *gstbasetransform_class = GST_BASE_TRANSFORM_CLASS(klass);

  gobject_class->dispose = gst_nvds_kafka_dispose;
  gobject_class->finalize = gst_nvds_kafka_finalize;
  gobject_class->set_property = gst_nvds_kafka_set_property;
  gobject_class->get_property = gst_nvds_kafka_get_property;

  gstbasetransform_class->start = gst_nvds_kafka_start;
  gstbasetransform_class->stop = gst_nvds_kafka_stop;
  gstbasetransform_class->transform_ip = gst_nvds_kafka_transform_ip;

  /* Properties */
  g_object_class_install_property(gobject_class, PROP_BROKERS,
      g_param_spec_string("brokers", "Kafka Brokers",
          "Comma-separated list of Kafka brokers (host:port)",
          DEFAULT_BROKERS, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(gobject_class, PROP_TOPIC,
      g_param_spec_string("topic", "Kafka Topic",
          "Kafka topic to publish messages to",
          DEFAULT_TOPIC, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(gobject_class, PROP_CLIENT_ID,
      g_param_spec_string("client-id", "Kafka Client ID",
          "Kafka client identifier",
          DEFAULT_CLIENT_ID, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(gobject_class, PROP_CONFIG_FILE,
      g_param_spec_string("config-file", "Config File",
          "Path to Kafka configuration file",
          NULL, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(gobject_class, PROP_CUSTOM_TEMPLATE,
      g_param_spec_string("custom-template", "Custom Template",
          "Path to custom JSON template file",
          NULL, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(gobject_class, PROP_MSG_FORMAT,
      g_param_spec_enum("msg-format", "Message Format",
          "JSON message format type",
          GST_TYPE_NVDS_KAFKA_MSG_FORMAT, DEFAULT_MSG_FORMAT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(gobject_class, PROP_ENABLE_DEBUG,
      g_param_spec_boolean("enable-debug", "Enable Debug",
          "Enable debug logging",
          FALSE, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(gobject_class, PROP_ASYNC_SEND,
      g_param_spec_boolean("async-send", "Async Send",
          "Enable asynchronous message sending",
          DEFAULT_ASYNC_SEND, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(gobject_class, PROP_BATCH_SIZE,
      g_param_spec_int("batch-size", "Batch Size",
          "Number of messages to batch before sending",
          1, 10000, DEFAULT_BATCH_SIZE, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(gobject_class, PROP_FLUSH_TIMEOUT,
      g_param_spec_int("flush-timeout", "Flush Timeout",
          "Timeout for flushing batched messages (milliseconds)",
          100, 60000, DEFAULT_FLUSH_TIMEOUT, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(gobject_class, PROP_INCLUDE_ALL_ATTRS,
      g_param_spec_boolean("include-all-attrs", "Include All Attributes",
          "Include all available metadata attributes in message",
          DEFAULT_INCLUDE_ALL_ATTRS, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(gobject_class, PROP_PARSE_OTHER_ATTRS,
      g_param_spec_boolean("parse-other-attrs", "Parse Other Attributes",
          "Parse otherAttrs field into separate JSON fields",
          DEFAULT_PARSE_OTHER_ATTRS, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(gobject_class, PROP_OTHER_ATTRS_SEPARATOR,
      g_param_spec_string("other-attrs-separator", "Other Attrs Separator",
          "Separator used in otherAttrs field",
          DEFAULT_OTHER_ATTRS_SEPARATOR, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property(gobject_class, PROP_KEY_VALUE_SEPARATOR,
      g_param_spec_string("key-value-separator", "Key-Value Separator",
          "Key-value separator used in otherAttrs field",
          DEFAULT_KEY_VALUE_SEPARATOR, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* Element metadata */
  gst_element_class_set_static_metadata(gstelement_class,
      "NVIDIA DeepStream Kafka Publisher",
      "Sink/Network",
      "Publishes DeepStream event metadata to Apache Kafka with full customization",
      "NVIDIA Corporation");

  gst_element_class_add_static_pad_template(gstelement_class, &gst_nvds_kafka_sink_template);
  gst_element_class_add_static_pad_template(gstelement_class, &gst_nvds_kafka_src_template);

  GST_DEBUG_CATEGORY_INIT(gst_nvds_kafka_debug, "nvdskafka", 0, "nvdskafka plugin");
}

static void
gst_nvds_kafka_init(GstNvdsKafka *nvds_kafka)
{
  /* Initialize properties with defaults */
  nvds_kafka->brokers = g_strdup(DEFAULT_BROKERS);
  nvds_kafka->topic = g_strdup(DEFAULT_TOPIC);
  nvds_kafka->client_id = g_strdup(DEFAULT_CLIENT_ID);
  nvds_kafka->config_file = NULL;
  nvds_kafka->custom_template = NULL;
  nvds_kafka->msg_format = DEFAULT_MSG_FORMAT;
  nvds_kafka->enable_debug = FALSE;
  nvds_kafka->async_send = DEFAULT_ASYNC_SEND;
  nvds_kafka->batch_size = DEFAULT_BATCH_SIZE;
  nvds_kafka->flush_timeout = DEFAULT_FLUSH_TIMEOUT;
  nvds_kafka->include_all_attrs = DEFAULT_INCLUDE_ALL_ATTRS;
  nvds_kafka->parse_other_attrs = DEFAULT_PARSE_OTHER_ATTRS;
  nvds_kafka->other_attrs_separator = g_strdup(DEFAULT_OTHER_ATTRS_SEPARATOR);
  nvds_kafka->key_value_separator = g_strdup(DEFAULT_KEY_VALUE_SEPARATOR);

  /* Initialize runtime data */
  nvds_kafka->kafka_producer = NULL;
  nvds_kafka->custom_json_template = NULL;
  g_mutex_init(&nvds_kafka->kafka_lock);
  nvds_kafka->message_count = 0;
  nvds_kafka->error_count = 0;
  nvds_kafka->is_initialized = FALSE;

  /* Set transform properties */
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(nvds_kafka), TRUE);
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(nvds_kafka), TRUE);
}

static void
gst_nvds_kafka_dispose(GObject *object)
{
  GstNvdsKafka *nvds_kafka = GST_NVDS_KAFKA(object);

  gst_nvds_kafka_cleanup_producer(nvds_kafka);

  G_OBJECT_CLASS(parent_class)->dispose(object);
}

static void
gst_nvds_kafka_finalize(GObject *object)
{
  GstNvdsKafka *nvds_kafka = GST_NVDS_KAFKA(object);

  g_free(nvds_kafka->brokers);
  g_free(nvds_kafka->topic);
  g_free(nvds_kafka->client_id);
  g_free(nvds_kafka->config_file);
  g_free(nvds_kafka->custom_template);
  g_free(nvds_kafka->other_attrs_separator);
  g_free(nvds_kafka->key_value_separator);

  if (nvds_kafka->custom_json_template) {
    delete nvds_kafka->custom_json_template;
  }

  g_mutex_clear(&nvds_kafka->kafka_lock);

  G_OBJECT_CLASS(parent_class)->finalize(object);
}

static void
gst_nvds_kafka_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec)
{
  GstNvdsKafka *nvds_kafka = GST_NVDS_KAFKA(object);

  switch (prop_id) {
    case PROP_BROKERS:
      g_free(nvds_kafka->brokers);
      nvds_kafka->brokers = g_value_dup_string(value);
      break;
    case PROP_TOPIC:
      g_free(nvds_kafka->topic);
      nvds_kafka->topic = g_value_dup_string(value);
      break;
    case PROP_CLIENT_ID:
      g_free(nvds_kafka->client_id);
      nvds_kafka->client_id = g_value_dup_string(value);
      break;
    case PROP_CONFIG_FILE:
      g_free(nvds_kafka->config_file);
      nvds_kafka->config_file = g_value_dup_string(value);
      break;
    case PROP_CUSTOM_TEMPLATE:
      g_free(nvds_kafka->custom_template);
      nvds_kafka->custom_template = g_value_dup_string(value);
      break;
    case PROP_MSG_FORMAT:
      nvds_kafka->msg_format = (GstNvdsKafkaMsgFormat)g_value_get_enum(value);
      break;
    case PROP_ENABLE_DEBUG:
      nvds_kafka->enable_debug = g_value_get_boolean(value);
      break;
    case PROP_ASYNC_SEND:
      nvds_kafka->async_send = g_value_get_boolean(value);
      break;
    case PROP_BATCH_SIZE:
      nvds_kafka->batch_size = g_value_get_int(value);
      break;
    case PROP_FLUSH_TIMEOUT:
      nvds_kafka->flush_timeout = g_value_get_int(value);
      break;
    case PROP_INCLUDE_ALL_ATTRS:
      nvds_kafka->include_all_attrs = g_value_get_boolean(value);
      break;
    case PROP_PARSE_OTHER_ATTRS:
      nvds_kafka->parse_other_attrs = g_value_get_boolean(value);
      break;
    case PROP_OTHER_ATTRS_SEPARATOR:
      g_free(nvds_kafka->other_attrs_separator);
      nvds_kafka->other_attrs_separator = g_value_dup_string(value);
      break;
    case PROP_KEY_VALUE_SEPARATOR:
      g_free(nvds_kafka->key_value_separator);
      nvds_kafka->key_value_separator = g_value_dup_string(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void
gst_nvds_kafka_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec)
{
  GstNvdsKafka *nvds_kafka = GST_NVDS_KAFKA(object);

  switch (prop_id) {
    case PROP_BROKERS:
      g_value_set_string(value, nvds_kafka->brokers);
      break;
    case PROP_TOPIC:
      g_value_set_string(value, nvds_kafka->topic);
      break;
    case PROP_CLIENT_ID:
      g_value_set_string(value, nvds_kafka->client_id);
      break;
    case PROP_CONFIG_FILE:
      g_value_set_string(value, nvds_kafka->config_file);
      break;
    case PROP_CUSTOM_TEMPLATE:
      g_value_set_string(value, nvds_kafka->custom_template);
      break;
    case PROP_MSG_FORMAT:
      g_value_set_enum(value, nvds_kafka->msg_format);
      break;
    case PROP_ENABLE_DEBUG:
      g_value_set_boolean(value, nvds_kafka->enable_debug);
      break;
    case PROP_ASYNC_SEND:
      g_value_set_boolean(value, nvds_kafka->async_send);
      break;
    case PROP_BATCH_SIZE:
      g_value_set_int(value, nvds_kafka->batch_size);
      break;
    case PROP_FLUSH_TIMEOUT:
      g_value_set_int(value, nvds_kafka->flush_timeout);
      break;
    case PROP_INCLUDE_ALL_ATTRS:
      g_value_set_boolean(value, nvds_kafka->include_all_attrs);
      break;
    case PROP_PARSE_OTHER_ATTRS:
      g_value_set_boolean(value, nvds_kafka->parse_other_attrs);
      break;
    case PROP_OTHER_ATTRS_SEPARATOR:
      g_value_set_string(value, nvds_kafka->other_attrs_separator);
      break;
    case PROP_KEY_VALUE_SEPARATOR:
      g_value_set_string(value, nvds_kafka->key_value_separator);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static gboolean
gst_nvds_kafka_start(GstBaseTransform *trans)
{
  GstNvdsKafka *nvds_kafka = GST_NVDS_KAFKA(trans);

  GST_DEBUG_OBJECT(nvds_kafka, "Starting nvdskafka plugin");

  if (!gst_nvds_kafka_init_producer(nvds_kafka)) {
    GST_ERROR_OBJECT(nvds_kafka, "Failed to initialize Kafka producer");
    return FALSE;
  }

  if (nvds_kafka->msg_format == GST_NVDS_KAFKA_MSG_FORMAT_CUSTOM) {
    if (!gst_nvds_kafka_load_custom_template(nvds_kafka)) {
      GST_WARNING_OBJECT(nvds_kafka, "Failed to load custom template, using full format");
      nvds_kafka->msg_format = GST_NVDS_KAFKA_MSG_FORMAT_FULL;
    }
  }

  nvds_kafka->is_initialized = TRUE;
  GST_INFO_OBJECT(nvds_kafka, "nvdskafka plugin started successfully");

  return TRUE;
}

static gboolean
gst_nvds_kafka_stop(GstBaseTransform *trans)
{
  GstNvdsKafka *nvds_kafka = GST_NVDS_KAFKA(trans);

  GST_DEBUG_OBJECT(nvds_kafka, "Stopping nvdskafka plugin");

  gst_nvds_kafka_cleanup_producer(nvds_kafka);
  nvds_kafka->is_initialized = FALSE;

  GST_INFO_OBJECT(nvds_kafka, "nvdskafka plugin stopped. Messages sent: %" G_GUINT64_FORMAT
                               ", Errors: %" G_GUINT64_FORMAT,
                               nvds_kafka->message_count, nvds_kafka->error_count);

  return TRUE;
}

static GstFlowReturn
gst_nvds_kafka_transform_ip(GstBaseTransform *trans, GstBuffer *buf)
{
  GstNvdsKafka *nvds_kafka = GST_NVDS_KAFKA(trans);
  NvDsBatchMeta *batch_meta = NULL;
  NvDsFrameMeta *frame_meta = NULL;
  NvDsUserMeta *user_meta = NULL;
  NvDsEventMsgMeta *event_meta = NULL;

  if (!nvds_kafka->is_initialized) {
    GST_WARNING_OBJECT(nvds_kafka, "Plugin not initialized");
    return GST_FLOW_OK;
  }

  batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  if (!batch_meta) {
    GST_DEBUG_OBJECT(nvds_kafka, "No batch metadata found");
    return GST_FLOW_OK;
  }

  /* Iterate through frame metadata */
  for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
    frame_meta = (NvDsFrameMeta *)(l_frame->data);

    /* Iterate through user metadata to find event messages */
    for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
      user_meta = (NvDsUserMeta *)(l_user->data);

      if (user_meta->base_meta.meta_type == NVDS_EVENT_MSG_META) {
        event_meta = (NvDsEventMsgMeta *)(user_meta->user_meta_data);

        if (event_meta) {
          Json::Value json_message;

          /* Generate JSON message based on format */
          switch (nvds_kafka->msg_format) {
            case GST_NVDS_KAFKA_MSG_FORMAT_MINIMAL:
              json_message = gst_nvds_kafka_create_minimal_message(nvds_kafka, event_meta);
              break;
            case GST_NVDS_KAFKA_MSG_FORMAT_FULL:
              json_message = gst_nvds_kafka_create_full_message(nvds_kafka, event_meta);
              break;
            case GST_NVDS_KAFKA_MSG_FORMAT_CUSTOM:
              json_message = gst_nvds_kafka_create_custom_message(nvds_kafka, event_meta);
              break;
            default:
              json_message = gst_nvds_kafka_create_full_message(nvds_kafka, event_meta);
              break;
          }

          /* Convert JSON to string and send to Kafka */
          Json::StreamWriterBuilder builder;
          builder["indentation"] = "";
          std::string json_string = Json::writeString(builder, json_message);

          /* Generate message key based on sensor ID and object ID */
          std::string message_key;
          if (event_meta->sensorStr) {
            message_key = std::string(event_meta->sensorStr);
            if (event_meta->objectId) { // Use objectId string from event_meta
              message_key += "_" + std::string(event_meta->objectId);
            }
          } else {
            message_key = gst_nvds_kafka_generate_uuid();
          }

          if (!gst_nvds_kafka_send_message(nvds_kafka, json_string.c_str(), message_key.c_str())) {
            GST_WARNING_OBJECT(nvds_kafka, "Failed to send message to Kafka");
            nvds_kafka->error_count++;
          } else {
            nvds_kafka->message_count++;
            if (nvds_kafka->enable_debug) {
              GST_DEBUG_OBJECT(nvds_kafka, "Sent message: %s", json_string.c_str());
            }
          }
        }
      }
    }
  }

  return GST_FLOW_OK;
}

/* Kafka producer initialization */
gboolean
gst_nvds_kafka_init_producer(GstNvdsKafka *nvds_kafka)
{
  rd_kafka_conf_t *conf;
  rd_kafka_topic_conf_t *topic_conf;
  char errstr[512];

  g_mutex_lock(&nvds_kafka->kafka_lock);

  /* Create Kafka configuration */
  conf = rd_kafka_conf_new();
  topic_conf = rd_kafka_topic_conf_new();

  /* Set basic configuration */
  if (rd_kafka_conf_set(conf, "bootstrap.servers", nvds_kafka->brokers, errstr, sizeof(errstr)) != RD_KAFKA_CONF_OK) {
    GST_ERROR_OBJECT(nvds_kafka, "Failed to set bootstrap.servers: %s", errstr);
    rd_kafka_conf_destroy(conf);
    rd_kafka_topic_conf_destroy(topic_conf);
    g_mutex_unlock(&nvds_kafka->kafka_lock);
    return FALSE;
  }

  if (rd_kafka_conf_set(conf, "client.id", nvds_kafka->client_id, errstr, sizeof(errstr)) != RD_KAFKA_CONF_OK) {
    GST_WARNING_OBJECT(nvds_kafka, "Failed to set client.id: %s", errstr);
  }

  /* Load additional configuration from file if specified */
  if (nvds_kafka->config_file) {
    std::ifstream config_file(nvds_kafka->config_file);
    std::string line;

    while (std::getline(config_file, line)) {
      if (line.empty() || line[0] == '#') continue;

      size_t pos = line.find('=');
      if (pos != std::string::npos) {
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        if (rd_kafka_conf_set(conf, key.c_str(), value.c_str(), errstr, sizeof(errstr)) != RD_KAFKA_CONF_OK) {
          GST_WARNING_OBJECT(nvds_kafka, "Failed to set config %s=%s: %s", key.c_str(), value.c_str(), errstr);
        }
      }
    }
  }

  /* Set performance optimizations */
  rd_kafka_conf_set(conf, "queue.buffering.max.messages", std::to_string(nvds_kafka->batch_size * 2).c_str(), NULL, 0);
  rd_kafka_conf_set(conf, "queue.buffering.max.ms", std::to_string(nvds_kafka->flush_timeout).c_str(), NULL, 0);
  rd_kafka_conf_set(conf, "batch.num.messages", std::to_string(nvds_kafka->batch_size).c_str(), NULL, 0);

  /* Create producer */
  nvds_kafka->kafka_producer = new KafkaProducer();
  nvds_kafka->kafka_producer->producer = rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));

  if (!nvds_kafka->kafka_producer->producer) {
    GST_ERROR_OBJECT(nvds_kafka, "Failed to create producer: %s", errstr);
    delete nvds_kafka->kafka_producer;
    nvds_kafka->kafka_producer = NULL;
    rd_kafka_topic_conf_destroy(topic_conf);
    g_mutex_unlock(&nvds_kafka->kafka_lock);
    return FALSE;
  }

  /* Create topic */
  nvds_kafka->kafka_producer->topic = rd_kafka_topic_new(nvds_kafka->kafka_producer->producer,
                                                          nvds_kafka->topic, topic_conf);
  if (!nvds_kafka->kafka_producer->topic) {
    GST_ERROR_OBJECT(nvds_kafka, "Failed to create topic: %s", rd_kafka_err2str(rd_kafka_last_error()));
    rd_kafka_destroy(nvds_kafka->kafka_producer->producer);
    delete nvds_kafka->kafka_producer;
    nvds_kafka->kafka_producer = NULL;
    g_mutex_unlock(&nvds_kafka->kafka_lock);
    return FALSE;
  }

  nvds_kafka->kafka_producer->is_connected = TRUE;
  g_mutex_unlock(&nvds_kafka->kafka_lock);

  GST_INFO_OBJECT(nvds_kafka, "Kafka producer initialized successfully");
  return TRUE;
}

/* Kafka producer cleanup */
void
gst_nvds_kafka_cleanup_producer(GstNvdsKafka *nvds_kafka)
{
  g_mutex_lock(&nvds_kafka->kafka_lock);

  if (nvds_kafka->kafka_producer) {
    if (nvds_kafka->kafka_producer->producer) {
      /* Flush any pending messages */
      rd_kafka_flush(nvds_kafka->kafka_producer->producer, 5000);

      if (nvds_kafka->kafka_producer->topic) {
        rd_kafka_topic_destroy(nvds_kafka->kafka_producer->topic);
        nvds_kafka->kafka_producer->topic = NULL;
      }

      rd_kafka_destroy(nvds_kafka->kafka_producer->producer);
      nvds_kafka->kafka_producer->producer = NULL;
    }

    delete nvds_kafka->kafka_producer;
    nvds_kafka->kafka_producer = NULL;
  }

  g_mutex_unlock(&nvds_kafka->kafka_lock);
}

/* Send message to Kafka */
gboolean
gst_nvds_kafka_send_message(GstNvdsKafka *nvds_kafka, const gchar *message, const gchar *key)
{
  if (!nvds_kafka->kafka_producer || !nvds_kafka->kafka_producer->producer) {
    return FALSE;
  }

  g_mutex_lock(&nvds_kafka->kafka_lock);

  int result = rd_kafka_produce(
      nvds_kafka->kafka_producer->topic,
      RD_KAFKA_PARTITION_UA,
      RD_KAFKA_MSG_F_COPY,
      (void *)message, strlen(message),
      key, key ? strlen(key) : 0,
      NULL);

  g_mutex_unlock(&nvds_kafka->kafka_lock);

  if (result == -1) {
    GST_ERROR_OBJECT(nvds_kafka, "Failed to produce message: %s",
                     rd_kafka_err2str(rd_kafka_last_error()));
    return FALSE;
  }

  /* Poll for events if async_send is disabled */
  if (!nvds_kafka->async_send) {
    rd_kafka_poll(nvds_kafka->kafka_producer->producer, 0);
  }

  return TRUE;
}

/* Generate UUID string */
std::string
gst_nvds_kafka_generate_uuid(void)
{
  uuid_t uuid;
  char uuid_str[37];

  uuid_generate(uuid);
  uuid_unparse(uuid, uuid_str);

  return std::string(uuid_str);
}

/* Parse otherAttrs field into JSON object */
void
gst_nvds_kafka_parse_other_attrs(GstNvdsKafka *nvds_kafka, const gchar *other_attrs, Json::Value &json_obj)
{
  if (!other_attrs || !nvds_kafka->parse_other_attrs) {
    return;
  }

  std::string attrs_str(other_attrs);
  std::string separator = nvds_kafka->other_attrs_separator;
  std::string kv_separator = nvds_kafka->key_value_separator;

  size_t pos = 0;
  while ((pos = attrs_str.find(separator)) != std::string::npos) {
    std::string pair = attrs_str.substr(0, pos);
    size_t kv_pos = pair.find(kv_separator);

    if (kv_pos != std::string::npos) {
      std::string key = pair.substr(0, kv_pos);
      std::string value = pair.substr(kv_pos + kv_separator.length());

      /* Try to parse as number, otherwise store as string */
      try {
        if (value.find('.') != std::string::npos) {
          json_obj[key] = std::stod(value);
        } else {
          // Explicitly cast to Int64 to resolve ambiguity
          json_obj[key] = Json::Value(static_cast<Json::Value::Int64>(std::stoll(value)));
        }
      } catch (...) {
        json_obj[key] = value;
      }
    }

    attrs_str.erase(0, pos + separator.length());
  }

  /* Handle the last pair */
  if (!attrs_str.empty()) {
    size_t kv_pos = attrs_str.find(kv_separator);
    if (kv_pos != std::string::npos) {
      std::string key = attrs_str.substr(0, kv_pos);
      std::string value = attrs_str.substr(kv_pos + kv_separator.length());

      try {
        if (value.find('.') != std::string::npos) {
          json_obj[key] = std::stod(value);
        } else {
          // Explicitly cast to Int64 to resolve ambiguity
          json_obj[key] = Json::Value(static_cast<Json::Value::Int64>(std::stoll(value)));
        }
      } catch (...) {
        json_obj[key] = value;
      }
    }
  }
}


/* Create minimal JSON message */
Json::Value
gst_nvds_kafka_create_minimal_message(GstNvdsKafka *nvds_kafka, NvDsEventMsgMeta *event_meta)
{
  Json::Value json_msg;

  /* Basic information */
  json_msg["messageid"] = gst_nvds_kafka_generate_uuid();
  // Use ts directly as string
  if (event_meta->ts) {
      json_msg["timestamp"] = event_meta->ts;
  }

  if (event_meta->sensorStr) {
    json_msg["sensorId"] = event_meta->sensorStr;
  }

  if (event_meta->type == NVDS_EVENT_ENTRY || event_meta->type == NVDS_EVENT_EXIT) {
    json_msg["type"] = (event_meta->type == NVDS_EVENT_ENTRY) ? "entry" : "exit";
  }

  /* Object information */
  if (event_meta->objectId) {
    json_msg["objectId"] = event_meta->objectId;
  }
  if (event_meta->sensorStr) { // Often used as object class name or related info
    json_msg["objectClass"] = event_meta->sensorStr; // Adjust if another field is better
  }

  return json_msg;
}


/* Create full JSON message */
Json::Value
gst_nvds_kafka_create_full_message(GstNvdsKafka *nvds_kafka, NvDsEventMsgMeta *event_meta)
{
  Json::Value json_msg;

  /* Message metadata */
  json_msg["messageid"] = gst_nvds_kafka_generate_uuid();
  json_msg["mdsversion"] = "1.0";
  // Use ts directly as string
  if (event_meta->ts) {
      json_msg["timestamp"] = event_meta->ts;
  }

  /* Sensor information */
  if (event_meta->sensorStr) {
    json_msg["sensorId"] = event_meta->sensorStr;
  }

  /* Event type */
  switch (event_meta->type) {
    case NVDS_EVENT_ENTRY:
      json_msg["@type"] = "entry";
      break;
    case NVDS_EVENT_EXIT:
      json_msg["@type"] = "exit";
      break;
    case NVDS_EVENT_MOVING:
      json_msg["@type"] = "moving";
      break;
    case NVDS_EVENT_STOPPED:
      json_msg["@type"] = "stopped";
      break;
    case NVDS_EVENT_EMPTY:
      json_msg["@type"] = "empty";
      break;
    case NVDS_EVENT_PARKED:
      json_msg["@type"] = "parked";
      break;
    case NVDS_EVENT_RESET:
      json_msg["@type"] = "reset";
      break;
    default:
      json_msg["@type"] = "unknown";
      break;
  }

  /* Object information - Use fields directly from event_meta */
  Json::Value object_info;
  if (event_meta->objectId) {
      object_info["id"] = event_meta->objectId;
  }
  object_info["speed"] = event_meta->confidence; // Map confidence to speed or use a fixed value if speed isn't available
  object_info["direction"] = 0.0; // Placeholder, direction not directly in event_meta
  object_info["orientation"] = 0.0; // Placeholder, orientation not directly in event_meta
  if (event_meta->sensorStr) {
      object_info["class"] = event_meta->sensorStr; // Map sensorStr or find a better class name source
  }
  object_info["confidence"] = event_meta->confidence;
  if (event_meta->trackingId > 0) {
      object_info["trackingId"] = Json::Value(static_cast<Json::Value::UInt64>(event_meta->trackingId));
  }

  /* Bounding box */
  Json::Value bbox;
  bbox["topleftx"] = event_meta->bbox.left;
  bbox["toplefty"] = event_meta->bbox.top;
  bbox["width"] = event_meta->bbox.width;
  bbox["height"] = event_meta->bbox.height;
  object_info["bbox"] = bbox;

  json_msg["object"] = object_info;


  /* Location information - Use NvDsGeoLocation */
  Json::Value location_obj;
  location_obj["lat"] = event_meta->location.lat;
  location_obj["lon"] = event_meta->location.lon;
  location_obj["alt"] = event_meta->location.alt;
  json_msg["location"] = location_obj;


  /* Coordinate information - Use NvDsCoordinate */
  Json::Value coord_obj;
  coord_obj["x"] = event_meta->coordinate.x;
  coord_obj["y"] = event_meta->coordinate.y;
  coord_obj["z"] = event_meta->coordinate.z;
  json_msg["coordinate"] = coord_obj;


  /* Video frame information */
  json_msg["frameId"] = event_meta->frameId; // Add frameId

  /* Additional attributes */
  if (nvds_kafka->include_all_attrs) {
    json_msg["objType"] = event_meta->objType;
    json_msg["objClassId"] = event_meta->objClassId;
    json_msg["sensorId"] = event_meta->sensorId; // Redundant with sensorStr?
    json_msg["placeId"] = event_meta->placeId;
    json_msg["moduleId"] = event_meta->moduleId;

    if (event_meta->extMsgSize > 0 && event_meta->extMsg) {
      json_msg["extMsg"] = event_meta->extMsg;
    }

    /* Parse otherAttrs if available */
    if (event_meta->otherAttrs) {
      if (nvds_kafka->parse_other_attrs) {
        Json::Value other_attrs_obj;
        gst_nvds_kafka_parse_other_attrs(nvds_kafka, event_meta->otherAttrs, other_attrs_obj);
        json_msg["otherAttrs"] = other_attrs_obj;
      } else {
        json_msg["otherAttrs"] = event_meta->otherAttrs;
      }
    }
  }

  return json_msg;
}


/* Create custom JSON message based on template */
Json::Value
gst_nvds_kafka_create_custom_message(GstNvdsKafka *nvds_kafka, NvDsEventMsgMeta *event_meta)
{
  if (!nvds_kafka->custom_json_template) {
    return gst_nvds_kafka_create_full_message(nvds_kafka, event_meta);
  }

  Json::Value json_msg = *nvds_kafka->custom_json_template;

  /* Replace template placeholders with actual values */
  std::function<void(Json::Value&)> replace_placeholders = [&](Json::Value& node) {
    if (node.isString()) {
      std::string str_val = node.asString();

      /* Replace common placeholders */
      if (str_val == "${messageid}") {
        node = gst_nvds_kafka_generate_uuid();
      } else if (str_val == "${timestamp}" && event_meta->ts) {
        node = event_meta->ts;
      } else if (str_val == "${sensorId}" && event_meta->sensorStr) {
        node = event_meta->sensorStr;
      } else if (str_val == "${objectId}" && event_meta->objectId) {
        node = event_meta->objectId;
      } else if (str_val == "${objectClass}" && event_meta->sensorStr) {
        node = event_meta->sensorStr; // Adjust if another field is better
      } else if (str_val == "${location}" && (event_meta->location.lat != 0.0 || event_meta->location.lon != 0.0)) {
          std::ostringstream loc_stream;
          loc_stream << std::fixed << std::setprecision(6) << event_meta->location.lat << ","
                     << std::fixed << std::setprecision(6) << event_meta->location.lon << ","
                     << std::fixed << std::setprecision(2) << event_meta->location.alt;
          node = loc_stream.str();
      }
      /* Add more placeholder replacements as needed */
    } else if (node.isObject()) {
      for (const auto& key : node.getMemberNames()) {
        replace_placeholders(node[key]);
      }
    } else if (node.isArray()) {
      for (Json::Value::ArrayIndex i = 0; i < node.size(); i++) {
        replace_placeholders(node[i]);
      }
    }
  };

  replace_placeholders(json_msg);
  return json_msg;
}


/* Load custom JSON template from file */
gboolean
gst_nvds_kafka_load_custom_template(GstNvdsKafka *nvds_kafka)
{
  if (!nvds_kafka->custom_template) {
    GST_WARNING_OBJECT(nvds_kafka, "No custom template file specified");
    return FALSE;
  }

  std::ifstream template_file(nvds_kafka->custom_template);
  if (!template_file.is_open()) {
    GST_ERROR_OBJECT(nvds_kafka, "Failed to open template file: %s", nvds_kafka->custom_template);
    return FALSE;
  }

  try {
    Json::CharReaderBuilder builder;
    std::string errors;

    nvds_kafka->custom_json_template = new Json::Value();

    if (!Json::parseFromStream(builder, template_file, nvds_kafka->custom_json_template, &errors)) {
      GST_ERROR_OBJECT(nvds_kafka, "Failed to parse JSON template: %s", errors.c_str());
      delete nvds_kafka->custom_json_template;
      nvds_kafka->custom_json_template = NULL;
      return FALSE;
    }

    GST_INFO_OBJECT(nvds_kafka, "Custom template loaded successfully from %s", nvds_kafka->custom_template);
    return TRUE;

  } catch (const std::exception& e) {
    GST_ERROR_OBJECT(nvds_kafka, "Exception while loading template: %s", e.what());
    if (nvds_kafka->custom_json_template) {
      delete nvds_kafka->custom_json_template;
      nvds_kafka->custom_json_template = NULL;
    }
    return FALSE;
  }
}

/* Plugin entry point */
static gboolean
nvds_kafka_plugin_init(GstPlugin *plugin)
{
  GST_DEBUG_CATEGORY_INIT(gst_nvds_kafka_debug, "nvdskafka", 0, "nvdskafka plugin");

  return gst_element_register(plugin, "nvdskafka", GST_RANK_PRIMARY, GST_TYPE_NVDS_KAFKA);
}

// If config.h is not available or doesn't define PACKAGE, you might need to define it here
// or ensure your build system (CMake) correctly generates it.
#ifndef PACKAGE
#define PACKAGE "nvdskafka"
#endif

GST_PLUGIN_DEFINE(
  GST_VERSION_MAJOR,
  GST_VERSION_MINOR,
  nvdskafka,
  "NVIDIA DeepStream Kafka Publisher Plugin with Full Customization",
  nvds_kafka_plugin_init,
  "1.0",
  "MIT", // Or use the SPDX identifier from the header if different
  "NVIDIA DeepStream Kafka Plugin",
  "https://developer.nvidia.com/deepstream-sdk"
)
