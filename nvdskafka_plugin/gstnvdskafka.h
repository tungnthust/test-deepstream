/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __GST_NVDS_KAFKA_H__
#define __GST_NVDS_KAFKA_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <librdkafka/rdkafka.h>
// Include DeepStream headers for NvDsEventMsgMeta, NvDsBatchMeta, etc.
#include "nvdsmeta.h"       // For NvDsBatchMeta, NvDsFrameMeta, NvDsUserMeta
#include "nvdsmeta_schema.h" // For NvDsEventMsgMeta

#include <jsoncpp/json/json.h>
#include <string>
// #include <memory> // Not strictly needed if we manage KafkaProducer manually
#include <mutex> // Required for std::mutex if used directly, otherwise rely on glib for GMutex

G_BEGIN_DECLS

/* Standard GObject macros */
#define GST_TYPE_NVDS_KAFKA (gst_nvds_kafka_get_type())
#define GST_NVDS_KAFKA(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVDS_KAFKA, GstNvdsKafka))
#define GST_NVDS_KAFKA_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NVDS_KAFKA, GstNvdsKafkaClass))
#define GST_NVDS_KAFKA_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_NVDS_KAFKA, GstNvdsKafkaClass))
#define GST_IS_NVDS_KAFKA(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVDS_KAFKA))
#define GST_IS_NVDS_KAFKA_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NVDS_KAFKA))
#define GST_NVDS_KAFKA_CAST(obj) ((GstNvdsKafka *)(obj))

typedef struct _GstNvdsKafka GstNvdsKafka;
typedef struct _GstNvdsKafkaClass GstNvdsKafkaClass;

/**
 * Message format options
 */
typedef enum {
  GST_NVDS_KAFKA_MSG_FORMAT_MINIMAL,     /**< Minimal JSON with essential fields */
  GST_NVDS_KAFKA_MSG_FORMAT_FULL,        /**< Full JSON with all available fields */
  GST_NVDS_KAFKA_MSG_FORMAT_CUSTOM       /**< Custom JSON format based on template */
} GstNvdsKafkaMsgFormat;

/**
 * Kafka producer configuration structure
 * Note: Using C-style pointers and GMutex for GStreamer plugin compatibility.
 *       std::mutex commented out as GMutex is used.
 */
typedef struct {
  rd_kafka_t *producer;
  rd_kafka_topic_t *topic;
  // rd_kafka_conf_t *conf; // Usually transferred to rd_kafka_new, not stored
  // rd_kafka_topic_conf_t *topic_conf; // Usually transferred to rd_kafka_topic_new, not stored
  // std::mutex producer_mutex; // Using GMutex from glib instead for consistency
  gboolean is_connected;
} KafkaProducer;

/**
 * GstNvdsKafka structure
 */
struct _GstNvdsKafka {
  GstBaseTransform parent;

  /* Properties */
  gchar *brokers;                    /**< Kafka broker connection string */
  gchar *topic;                      /**< Kafka topic name */
  gchar *client_id;                  /**< Kafka client ID */
  gchar *config_file;                /**< Optional config file for advanced settings */
  gchar *custom_template;            /**< Custom JSON template file path */
  GstNvdsKafkaMsgFormat msg_format;  /**< Message format type */
  gboolean enable_debug;             /**< Enable debug logging */
  gboolean async_send;               /**< Enable asynchronous sending */
  gint batch_size;                   /**< Number of messages to batch */
  gint flush_timeout;                /**< Timeout for flushing messages (ms) */
  gboolean include_all_attrs;        /**< Include all metadata attributes */
  gboolean parse_other_attrs;        /**< Parse otherAttrs field */
  gchar *other_attrs_separator;      /**< Separator for otherAttrs parsing */
  gchar *key_value_separator;        /**< Key-value separator for otherAttrs */

  /* Runtime data */
  KafkaProducer *kafka_producer;
  Json::Value *custom_json_template;
  GMutex kafka_lock; // Using GMutex
  guint64 message_count;
  guint64 error_count;
  gboolean is_initialized;
};

struct _GstNvdsKafkaClass {
  GstBaseTransformClass parent_class;
};

/* Function declarations */
GType gst_nvds_kafka_get_type(void);

/* Kafka-related functions */
gboolean gst_nvds_kafka_init_producer(GstNvdsKafka *nvds_kafka);
void gst_nvds_kafka_cleanup_producer(GstNvdsKafka *nvds_kafka);
gboolean gst_nvds_kafka_send_message(GstNvdsKafka *nvds_kafka, const gchar *message, const gchar *key);

/* JSON generation functions - Updated signatures to use NvDsEventMsgMeta* */
Json::Value gst_nvds_kafka_create_minimal_message(GstNvdsKafka *nvds_kafka, NvDsEventMsgMeta *event_meta);
Json::Value gst_nvds_kafka_create_full_message(GstNvdsKafka *nvds_kafka, NvDsEventMsgMeta *event_meta);
Json::Value gst_nvds_kafka_create_custom_message(GstNvdsKafka *nvds_kafka, NvDsEventMsgMeta *event_meta);

/* Utility functions */
std::string gst_nvds_kafka_generate_uuid(void);
void gst_nvds_kafka_parse_other_attrs(GstNvdsKafka *nvds_kafka, const gchar *other_attrs, Json::Value &json_obj);
gboolean gst_nvds_kafka_load_custom_template(GstNvdsKafka *nvds_kafka);

G_END_DECLS

#endif /* __GST_NVDS_KAFKA_H__ */
