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

// At the very top of gstnvdszonefilter.cpp
#include "gstnvdszonefilter.h"
#include "nvdszonefilter_property_parser.h"
#include <math.h>
#include <string.h>
#include <unordered_set> // Include for std::unordered_set
#include <vector>        // Include for std::vector

// Standard way, as in nvdsanalytics
GST_DEBUG_CATEGORY(gst_nvdszonefilter_debug);
#define GST_CAT_DEFAULT gst_nvdszonefilter_debug

/* Enum to identify properties */
enum {
    PROP_0,
    PROP_UNIQUE_ID,
    PROP_CONFIG_FILE,
    PROP_ENABLE,
    PROP_DEBOUNCE_FRAMES, // Add new property ID
    PROP_DRAW_ZONES       // Add zone drawing property
};

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 18 // Different from analytics
#define DEFAULT_DEBOUNCE_FRAMES 30 // Default debounce threshold (changed from 150 to fit within 0-100 range)
#define DEFAULT_DRAW_ZONES FALSE    // Default zone drawing disabled
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"

/* Zone drawing colors (RGBA format) */
#define ZONE_COLOR_RED    {1.0f, 0.0f, 0.0f, 0.8f}  // Semi-transparent red
#define ZONE_COLOR_GREEN  {0.0f, 1.0f, 0.0f, 0.8f}  // Semi-transparent green
#define ZONE_COLOR_BLUE   {0.0f, 0.0f, 1.0f, 0.8f}  // Semi-transparent blue
#define ZONE_COLOR_YELLOW {1.0f, 1.0f, 0.0f, 0.8f}  // Semi-transparent yellow

/* --- Pad Templates --- */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA }")));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA }")));

/* --- GType Definition --- */
#define gst_nvdszonefilter_parent_class parent_class
G_DEFINE_TYPE(GstNvDsZoneFilter, gst_nvdszonefilter, GST_TYPE_BASE_TRANSFORM);

/* --- Function Prototypes --- */
static void gst_nvdszonefilter_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec);
static void gst_nvdszonefilter_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec);
static gboolean gst_nvdszonefilter_set_caps(GstBaseTransform *btrans, GstCaps *incaps, GstCaps *outcaps);
static gboolean gst_nvdszonefilter_start(GstBaseTransform *btrans);
static gboolean gst_nvdszonefilter_stop(GstBaseTransform *btrans);
static GstFlowReturn gst_nvdszonefilter_transform_ip(GstBaseTransform *btrans, GstBuffer *inbuf);
static void gst_nvdszonefilter_finalize(GObject *object);

/* Helper function to create and populate NvDsEventMsgMeta */
static NvDsEventMsgMeta* create_zone_event_meta(GstNvDsZoneFilter *filter,
                                                NvDsEventType event_type, // Added event_type parameter
                                                gint stream_id,
                                                guint64 object_id,
                                                gdouble bbox_left, gdouble bbox_top,
                                                gdouble bbox_width, gdouble bbox_height,
                                                guint frame_width, guint frame_height,
                                                GstClockTime timestamp);

/* Helper function to draw zone overlay */
static void draw_zone_overlay(GstNvDsZoneFilter *filter, NvDsFrameMeta *frame_meta, 
                              StreamZoneConfig *zone_config, gint stream_id);

/* --- Class Initialization --- */
static void
gst_nvdszonefilter_class_init(GstNvDsZoneFilterClass *klass)
{
    GObjectClass *gobject_class = (GObjectClass *)klass;
    GstElementClass *gstelement_class = (GstElementClass *)klass;
    GstBaseTransformClass *gstbasetransform_class = (GstBaseTransformClass *)klass;

    /* Set metadata */
    gst_element_class_set_details_simple(gstelement_class,
        "DsZoneFilter Plugin",
        "Filter/Video",
        "Filters objects based on predefined zones using NvDsEventMsgMeta (Entry only once with debouncing)",
        "Your Name / NVIDIA");

    /* Override virtual methods */
    gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_nvdszonefilter_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_nvdszonefilter_get_property);
    gobject_class->finalize = GST_DEBUG_FUNCPTR(gst_nvdszonefilter_finalize);

    /* Install properties */
    g_object_class_install_property(gobject_class, PROP_UNIQUE_ID,
        g_param_spec_uint("unique-id", "Unique ID",
            "Unique ID for the element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_CONFIG_FILE,
        g_param_spec_string("config-file-path", "Zone Config File",
            "Path to the zone configuration INI file",
            NULL, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_ENABLE,
        g_param_spec_boolean("enable", "Enable",
            "Enable zone filtering, or set in passthrough mode",
            TRUE, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    // Install the debounce property
    g_object_class_install_property(gobject_class, PROP_DEBOUNCE_FRAMES,
        g_param_spec_uint("debounce-frames", "Debounce Frames",
            "Number of frames an object must be absent from the zone to trigger a new ENTRY event",
            0, 1000, DEFAULT_DEBOUNCE_FRAMES, // Min 0 (no debounce), Max 1000, Default 30
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    // Install the zone drawing property
    g_object_class_install_property(gobject_class, PROP_DRAW_ZONES,
        g_param_spec_boolean("draw-zones", "Draw Zones",
            "Enable drawing zone boundaries on the output frames",
            DEFAULT_DRAW_ZONES, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR(gst_nvdszonefilter_set_caps);
    gstbasetransform_class->start = GST_DEBUG_FUNCPTR(gst_nvdszonefilter_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR(gst_nvdszonefilter_stop);
    gstbasetransform_class->transform_ip = GST_DEBUG_FUNCPTR(gst_nvdszonefilter_transform_ip);

    /* Set pad templates */
    gst_element_class_add_pad_template(gstelement_class, gst_static_pad_template_get(&src_template));
    gst_element_class_add_pad_template(gstelement_class, gst_static_pad_template_get(&sink_template));
}

/* --- Instance Initialization --- */
static void
gst_nvdszonefilter_init(GstNvDsZoneFilter *filter)
{
    GST_DEBUG_OBJECT(filter, "Initializing GstNvDsZoneFilter");
    filter->unique_id = DEFAULT_UNIQUE_ID;
    filter->config_file_path = NULL;
    filter->config_file_parse_successful = FALSE;
    filter->enable = TRUE;
    filter->debounce_frame_count = DEFAULT_DEBOUNCE_FRAMES; // Initialize debounce
    filter->draw_zones = DEFAULT_DRAW_ZONES; // Initialize zone drawing
    filter->stream_zones_map = new std::unordered_map<gint, StreamZoneConfig>();
    // Initialize the new tracking map
    filter->tracked_objects_info = new std::unordered_map<gint, std::unordered_map<guint64, TrackedObjectInfo>>();
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(filter), TRUE);
    gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(filter), FALSE);
    g_mutex_init(&filter->config_mutex);
}

/* --- Finalize --- */
static void
gst_nvdszonefilter_finalize(GObject *object)
{
    GstNvDsZoneFilter *filter = GST_NVDSZONEFILTER(object);

    GST_DEBUG_OBJECT(filter, "Finalizing");

    g_free(filter->config_file_path);
    filter->config_file_path = NULL;

    if (filter->stream_zones_map) {
        filter->stream_zones_map->clear();
        delete filter->stream_zones_map;
        filter->stream_zones_map = NULL;
    }

    // Clean up tracked objects info map
    if (filter->tracked_objects_info) {
        // Iterate through each stream's map and clear it
        for (auto& stream_pair : *(filter->tracked_objects_info)) {
            stream_pair.second.clear(); // Clear the map of object info for this stream
        }
        filter->tracked_objects_info->clear(); // Clear the main map itself
        delete filter->tracked_objects_info;
        filter->tracked_objects_info = NULL;
    }

    g_mutex_clear(&filter->config_mutex);

    G_OBJECT_CLASS(parent_class)->finalize(object);
}

/* --- Property Setters/Getters --- */
static void
gst_nvdszonefilter_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec)
{
    GstNvDsZoneFilter *filter = GST_NVDSZONEFILTER(object);

    switch (prop_id) {
        case PROP_UNIQUE_ID:
            filter->unique_id = g_value_get_uint(value);
            break;
        case PROP_CONFIG_FILE: {
            g_mutex_lock(&filter->config_mutex);
            g_free(filter->config_file_path);
            filter->config_file_path = g_value_dup_string(value);
            filter->config_file_parse_successful = FALSE; // Reset flag

            if (filter->config_file_path && strlen(filter->config_file_path) > 0) {
                filter->config_file_parse_successful = nvdszonefilter_parse_config_file(filter, filter->config_file_path);
            } else {
                GST_WARNING_OBJECT(filter, "Config file path is empty");
            }
            g_mutex_unlock(&filter->config_mutex);
            break;
        }
        case PROP_ENABLE:
            filter->enable = g_value_get_boolean(value);
            break;
        case PROP_DEBOUNCE_FRAMES: // Handle debounce property
            filter->debounce_frame_count = g_value_get_uint(value);
            GST_INFO_OBJECT(filter, "Debounce frames set to %u", filter->debounce_frame_count);
            break;
        case PROP_DRAW_ZONES: // Handle zone drawing property
            filter->draw_zones = g_value_get_boolean(value);
            GST_INFO_OBJECT(filter, "Zone drawing %s", filter->draw_zones ? "enabled" : "disabled");
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void
gst_nvdszonefilter_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec)
{
    GstNvDsZoneFilter *filter = GST_NVDSZONEFILTER(object);

    switch (prop_id) {
        case PROP_UNIQUE_ID:
            g_value_set_uint(value, filter->unique_id);
            break;
        case PROP_CONFIG_FILE:
            g_value_set_string(value, filter->config_file_path);
            break;
        case PROP_ENABLE:
            g_value_set_boolean(value, filter->enable);
            break;
        case PROP_DEBOUNCE_FRAMES: // Handle debounce property
            g_value_set_uint(value, filter->debounce_frame_count);
            break;
        case PROP_DRAW_ZONES: // Handle zone drawing property
            g_value_set_boolean(value, filter->draw_zones);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

/* --- Lifecycle Methods --- */
static gboolean
gst_nvdszonefilter_set_caps(GstBaseTransform *btrans, GstCaps *incaps, GstCaps *outcaps)
{
    GstNvDsZoneFilter *filter = GST_NVDSZONEFILTER(btrans);
    GST_DEBUG_OBJECT(filter, "Caps set");
    return TRUE;
}

static gboolean
gst_nvdszonefilter_start(GstBaseTransform *btrans)
{
    GstNvDsZoneFilter *filter = GST_NVDSZONEFILTER(btrans);
    GST_DEBUG_OBJECT(filter, "Starting");
    GST_INFO_OBJECT(filter, "Debounce frames configured: %u", filter->debounce_frame_count);
    GST_INFO_OBJECT(filter, "Zone drawing: %s", filter->draw_zones ? "enabled" : "disabled");
    printf("Start Config file path: %s\n", filter->config_file_path);

    if (!filter->config_file_path || strlen(filter->config_file_path) == 0) {
        GST_ELEMENT_ERROR(filter, RESOURCE, SETTINGS, ("Configuration file path not provided"), (nullptr));
        return FALSE;
    }

    if (filter->config_file_parse_successful == FALSE) {
        GST_ELEMENT_ERROR(filter, RESOURCE, SETTINGS, ("Configuration file parsing failed"),
            ("Config file path: %s", filter->config_file_path));
        return FALSE;
    }

    return TRUE;
}

static gboolean
gst_nvdszonefilter_stop(GstBaseTransform *btrans)
{
    GstNvDsZoneFilter *filter = GST_NVDSZONEFILTER(btrans);
    GST_DEBUG_OBJECT(filter, "Stopping");

    // Optional: Clear tracked object state on stop. Depends on desired behavior.
    // If you want state to persist across pipeline PAUSE/PLAY, don't clear here.
    // If you want fresh state on every PLAY, clear here.
    g_mutex_lock(&filter->config_mutex);
    if (filter->tracked_objects_info) {
        for (auto& stream_pair : *(filter->tracked_objects_info)) {
            stream_pair.second.clear();
        }
        filter->tracked_objects_info->clear();
        GST_DEBUG_OBJECT(filter, "Cleared tracked object state on stop.");
    }
    g_mutex_unlock(&filter->config_mutex);

    return TRUE;
}

/* --- Core Processing --- */
/**
 * @brief Checks if the center of a bounding box is inside a normalized zone.
 */
static gboolean
is_object_in_zone(gdouble bbox_left, gdouble bbox_top, gdouble bbox_width, gdouble bbox_height,
                  gdouble zone_x1, gdouble zone_y1, gdouble zone_x2, gdouble zone_y2,
                  guint frame_width, guint frame_height)
{
    gdouble center_x = bbox_left + bbox_width / 2.0;
    gdouble center_y = bbox_top + bbox_height / 2.0;

    gdouble norm_center_x = center_x / frame_width;
    gdouble norm_center_y = center_y / frame_height;

    if (norm_center_x >= zone_x1 && norm_center_x <= zone_x2 &&
        norm_center_y >= zone_y1 && norm_center_y <= zone_y2) {
        return TRUE;
    }
    return FALSE;
}

/**
 * @brief Draw zone overlay using NVOSD
 */
static void
draw_zone_overlay(GstNvDsZoneFilter *filter, NvDsFrameMeta *frame_meta, 
                  StreamZoneConfig *zone_config, gint stream_id)
{
    if (!filter->draw_zones || !zone_config) {
        return;
    }

    // Get frame dimensions
    guint frame_width = frame_meta->source_frame_width;
    guint frame_height = frame_meta->source_frame_height;

    // Convert normalized coordinates to pixel coordinates
    guint zone_left = (guint)(zone_config->x1 * frame_width);
    guint zone_top = (guint)(zone_config->y1 * frame_height);
    guint zone_width = (guint)((zone_config->x2 - zone_config->x1) * frame_width);
    guint zone_height = (guint)((zone_config->y2 - zone_config->y1) * frame_height);

    // Create display meta for drawing the zone rectangle
    NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(frame_meta->base_meta.batch_meta);
    if (!display_meta) {
        GST_WARNING_OBJECT(filter, "Failed to acquire display meta from pool for stream %d", stream_id);
        return;
    }

    // Set up rectangle parameters
    NvOSD_RectParams *rect_params = &display_meta->rect_params[display_meta->num_rects];
    rect_params->left = zone_left;
    rect_params->top = zone_top;
    rect_params->width = zone_width;
    rect_params->height = zone_height;
    rect_params->border_width = 3; // Border thickness in pixels
    
    // Set zone color (use different colors for different streams if needed)
    switch (stream_id % 4) {
        case 0:
            rect_params->border_color.red = 1.0f;
            rect_params->border_color.green = 0.0f;
            rect_params->border_color.blue = 0.0f;
            rect_params->border_color.alpha = 0.8f;
            break;
        case 1:
            rect_params->border_color.red = 0.0f;
            rect_params->border_color.green = 1.0f;
            rect_params->border_color.blue = 0.0f;
            rect_params->border_color.alpha = 0.8f;
            break;
        case 2:
            rect_params->border_color.red = 0.0f;
            rect_params->border_color.green = 0.0f;
            rect_params->border_color.blue = 1.0f;
            rect_params->border_color.alpha = 0.8f;
            break;
        default:
            rect_params->border_color.red = 1.0f;
            rect_params->border_color.green = 1.0f;
            rect_params->border_color.blue = 0.0f;
            rect_params->border_color.alpha = 0.8f;
            break;
    }

    // Optional: Add semi-transparent fill
    rect_params->has_bg_color = 1;
    rect_params->bg_color = rect_params->border_color;
    rect_params->bg_color.alpha = 0.2f; // More transparent fill

    display_meta->num_rects++;

    // Add display meta to frame
    nvds_add_display_meta_to_frame(frame_meta, display_meta);

    GST_LOG_OBJECT(filter, "Drew zone overlay for stream %d: [%u, %u, %u, %u]", 
                   stream_id, zone_left, zone_top, zone_width, zone_height);
}

/**
 * @brief Creates and populates NvDsEventMsgMeta for zone detection event
 */
static NvDsEventMsgMeta*
create_zone_event_meta(GstNvDsZoneFilter *filter,
                       NvDsEventType event_type, // Added parameter
                       gint stream_id,
                       guint64 object_id,
                       gdouble bbox_left, gdouble bbox_top,
                       gdouble bbox_width, gdouble bbox_height,
                       guint frame_width, guint frame_height,
                       GstClockTime timestamp)
{
    NvDsEventMsgMeta *event_meta = (NvDsEventMsgMeta *)g_malloc0(sizeof(NvDsEventMsgMeta));

    // Basic event information - Use the passed event_type
    event_meta->type = event_type;
    event_meta->objType = NVDS_OBJECT_TYPE_UNKNOWN; // Can be modified based on object classification
    event_meta->confidence = 1.0f;
    event_meta->trackingId = object_id;
    event_meta->ts = g_strdup_printf("%" G_GUINT64_FORMAT, timestamp);

    // Sensor information
    event_meta->sensorId = stream_id;
    event_meta->placeId = filter->unique_id;
    event_meta->moduleId = filter->unique_id;
    event_meta->sensorStr = g_strdup_printf("sensor_%d", stream_id);

    // Object information
    event_meta->objectId = g_strdup_printf("%" G_GUINT64_FORMAT, object_id);

    // Bounding box information
    event_meta->bbox.top = bbox_top;
    event_meta->bbox.left = bbox_left;
    event_meta->bbox.width = bbox_width;
    event_meta->bbox.height = bbox_height;

    // Location information (normalized coordinates)
    event_meta->location.lat = (bbox_top + bbox_height / 2.0) / frame_height;
    event_meta->location.lon = (bbox_left + bbox_width / 2.0) / frame_width;
    event_meta->location.alt = 0.0;

    // Coordinate information (pixel coordinates)
    event_meta->coordinate.x = bbox_left + bbox_width / 2.0;
    event_meta->coordinate.y = bbox_top + bbox_height / 2.0;
    event_meta->coordinate.z = 0.0;

    // Additional metadata
    event_meta->otherAttrs = g_strdup_printf("zone_filter_id=%u;frame_width=%u;frame_height=%u;event_type=%s;debounce_frames=%u;draw_zones=%s",
                                           filter->unique_id, frame_width, frame_height,
                                           (event_type == NVDS_EVENT_ENTRY) ? "ENTRY" : "EXIT",
                                           filter->debounce_frame_count, 
                                           filter->draw_zones ? "true" : "false"); // Add drawing status to attrs

    GST_DEBUG("Created %s event meta for object %" G_GUINT64_FORMAT " in zone (stream %d)",
              (event_type == NVDS_EVENT_ENTRY) ? "ENTRY" : "EXIT", object_id, stream_id);

    return event_meta;
}

/**
 * @brief Copy function for NvDsEventMsgMeta
 */
static gpointer
copy_nvds_event_msg_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *src_user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *src_event_meta = (NvDsEventMsgMeta *)src_user_meta->user_meta_data;

    NvDsEventMsgMeta *dst_event_meta = (NvDsEventMsgMeta *)g_malloc0(sizeof(NvDsEventMsgMeta));

    // Copy basic fields
    dst_event_meta->type = src_event_meta->type;
    dst_event_meta->objType = src_event_meta->objType;
    dst_event_meta->confidence = src_event_meta->confidence;
    dst_event_meta->trackingId = src_event_meta->trackingId;
    dst_event_meta->sensorId = src_event_meta->sensorId;
    dst_event_meta->placeId = src_event_meta->placeId;
    dst_event_meta->moduleId = src_event_meta->moduleId;
    dst_event_meta->bbox = src_event_meta->bbox;
    dst_event_meta->location = src_event_meta->location;
    dst_event_meta->coordinate = src_event_meta->coordinate;

    // Copy string fields
    if (src_event_meta->ts)
        dst_event_meta->ts = g_strdup(src_event_meta->ts);
    if (src_event_meta->sensorStr)
        dst_event_meta->sensorStr = g_strdup(src_event_meta->sensorStr);
    if (src_event_meta->objectId)
        dst_event_meta->objectId = g_strdup(src_event_meta->objectId);
    if (src_event_meta->otherAttrs)
        dst_event_meta->otherAttrs = g_strdup(src_event_meta->otherAttrs);

    return dst_event_meta;
}

/**
 * @brief Release function for NvDsEventMsgMeta
 */
static void
release_nvds_event_msg_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *event_meta = (NvDsEventMsgMeta *)user_meta->user_meta_data;

    if (event_meta) {
        g_free(event_meta->ts);
        g_free(event_meta->sensorStr);
        g_free(event_meta->objectId);
        g_free(event_meta->otherAttrs);
        g_free(event_meta);
        user_meta->user_meta_data = NULL;
    }
}

static GstFlowReturn
gst_nvdszonefilter_transform_ip(GstBaseTransform *btrans, GstBuffer *inbuf)
{
    GstNvDsZoneFilter *filter = GST_NVDSZONEFILTER(btrans);

    if (!filter->enable) {
        GST_DEBUG_OBJECT(filter, "Plugin is disabled, passing through buffer.");
        return GST_FLOW_OK;
    }

    if (!filter->config_file_parse_successful) {
        GST_ELEMENT_ERROR(filter, RESOURCE, SETTINGS, ("Configuration not successfully parsed"), (nullptr));
        return GST_FLOW_ERROR;
    }

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(inbuf);
    if (!batch_meta) {
        GST_WARNING_OBJECT(filter, "NvDsBatchMeta not found!");
        return GST_FLOW_OK;
    }

    // Get buffer timestamp and frame number
    GstClockTime timestamp = GST_BUFFER_PTS(inbuf);
    if (!GST_CLOCK_TIME_IS_VALID(timestamp)) {
        timestamp = GST_CLOCK_TIME_NONE;
    }

    // Acquire meta lock for thread safety
    nvds_acquire_meta_lock(batch_meta);

    NvDsMetaList *l_frame = NULL;
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        guint64 current_frame_num = frame_meta->frame_num; // Use frame number from meta

        gint stream_id = frame_meta->pad_index;
        guint frame_width = frame_meta->source_frame_width;
        guint frame_height = frame_meta->source_frame_height;

        // Find the zone configuration for this stream
        StreamZoneConfig *zone_config = nullptr;
        g_mutex_lock(&filter->config_mutex); // Lock for accessing config and tracked_objects
        auto it = filter->stream_zones_map->find(stream_id);
        if (it != filter->stream_zones_map->end() && it->second.valid) {
            zone_config = &(it->second);
        }

        // Get or create the map of tracked object info for this stream
        std::unordered_map<guint64, TrackedObjectInfo>& tracked_info_map = (*(filter->tracked_objects_info))[stream_id];
        // No need to explicitly create the map, operator[] does it if key doesn't exist.

        g_mutex_unlock(&filter->config_mutex); // Unlock config_mutex after getting config and set reference

        if (!zone_config) {
            GST_DEBUG_OBJECT(filter, "No valid zone configuration found for stream %d, skipping frame processing.", stream_id);
            continue;
        }

        // Draw zone overlay ONLY for this stream's zone configuration
        draw_zone_overlay(filter, frame_meta, zone_config, stream_id);

        // Temporary set to store object IDs currently in the zone for this frame
        std::unordered_set<guint64> current_ids_in_zone;

        // Iterate through objects in the frame
        NvDsMetaList *l_obj = NULL;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            guint64 object_id = obj_meta->object_id; // Get object ID from tracker

            gboolean is_in_zone = is_object_in_zone(obj_meta->rect_params.left, obj_meta->rect_params.top,
                                  obj_meta->rect_params.width, obj_meta->rect_params.height,
                                  zone_config->x1, zone_config->y1, zone_config->x2, zone_config->y2,
                                  frame_width, frame_height);

            if (is_in_zone) {
                current_ids_in_zone.insert(object_id); // Add to current frame's list

                // Check the tracked info for this object
                auto tracked_it = tracked_info_map.find(object_id);
                if (tracked_it == tracked_info_map.end()) {
                    // Object is inside zone AND not previously tracked -> ENTRY EVENT
                    // Add to tracked map with current frame number
                    tracked_info_map[object_id] = {current_frame_num};

                    // Create NvDsEventMsgMeta for this zone ENTRY
                    NvDsEventMsgMeta *event_meta = create_zone_event_meta(filter,
                        NVDS_EVENT_ENTRY, // Specify ENTRY event type
                        stream_id, object_id,
                        obj_meta->rect_params.left, obj_meta->rect_params.top,
                        obj_meta->rect_params.width, obj_meta->rect_params.height,
                        frame_width, frame_height, timestamp);

                    // Acquire user meta from pool
                    NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
                    if (user_meta) {
                        user_meta->user_meta_data = (void *)event_meta;
                        user_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
                        user_meta->base_meta.copy_func = copy_nvds_event_msg_meta;
                        user_meta->base_meta.release_func = release_nvds_event_msg_meta;

                        // Add to frame metadata
                        nvds_add_user_meta_to_frame(frame_meta, user_meta);

                        GST_LOG_OBJECT(filter, "Added zone ENTRY event meta for object %" G_GUINT64_FORMAT " in stream %d (Frame %lu)",
                                       object_id, stream_id, current_frame_num);
                        printf("Added zone ENTRY event meta for object %" G_GUINT64_FORMAT " in stream %d (Frame %lu)\n",
                               object_id, stream_id, current_frame_num);
                    } else {
                        GST_WARNING_OBJECT(filter, "Failed to acquire user meta from pool for stream %d", stream_id);
                        // Clean up the event meta we created
                        if (event_meta) {
                             g_free(event_meta->ts);
                             g_free(event_meta->sensorStr);
                             g_free(event_meta->objectId);
                             g_free(event_meta->otherAttrs);
                             g_free(event_meta);
                        }
                    }
                } else {
                    // Object is inside zone AND was already tracked -> NO EVENT
                    // Update the last seen frame number
                    tracked_it->second.last_seen_frame_num = current_frame_num;
                    GST_LOG_OBJECT(filter, "Object %" G_GUINT64_FORMAT " already tracked in zone for stream %d (Frame %lu)", object_id, stream_id, current_frame_num);
                }
            }
            // If !is_in_zone, we don't do anything here in the loop for that object.
            // We handle objects that have potentially left the zone after processing all objects in the frame.
        } // End of object loop

        // --- Handle objects that might have LEFT the zone ---
        // Iterate through the map of tracked object info for this stream
        // We need to collect keys to remove to avoid iterator invalidation
        std::vector<guint64> ids_to_remove; // Collect object IDs that have left the zone long enough
        for (const auto& tracked_pair : tracked_info_map) {
            guint64 tracked_object_id = tracked_pair.first;
            guint64 last_seen_frame = tracked_pair.second.last_seen_frame_num;

            // If a tracked object ID is NOT in the current frame's list, it means it's not in the zone this frame
            if (current_ids_in_zone.find(tracked_object_id) == current_ids_in_zone.end()) {
                // Check if the debounce threshold has been exceeded
                // If the difference between current frame and last seen frame is greater than debounce threshold
                if ((current_frame_num >= last_seen_frame) && ((current_frame_num - last_seen_frame) > filter->debounce_frame_count)) {
                     // Object has been out of zone for longer than debounce period
                     ids_to_remove.push_back(tracked_object_id);
                     GST_LOG_OBJECT(filter, "Object %" G_GUINT64_FORMAT " removed from tracked set for stream %d (left zone for %lu frames, debounce=%u)",
                                    tracked_object_id, stream_id, (current_frame_num - last_seen_frame), filter->debounce_frame_count);
                     // Optional: Generate EXIT event here if needed
                     /*
                     // Find object meta if needed for EXIT event details (or use stored data)
                     // Create NvDsEventMsgMeta for EXIT event...
                     // Attach to frame meta...
                     */
                } else {
                    // Object is out of zone but within debounce window, keep it tracked
                    GST_LOG_OBJECT(filter, "Object %" G_GUINT64_FORMAT " out of zone but within debounce window (absent for %lu frames, debounce=%u)",
                                   tracked_object_id, (current_frame_num - last_seen_frame), filter->debounce_frame_count);
                }
            }
            // If the object IS in current_ids_in_zone, its last_seen_frame was already updated in the object loop.
        }

        // Remove the IDs that have definitively left the zone from the tracked map
        g_mutex_lock(&filter->config_mutex); // Re-lock before modifying tracked_info_map
        for (const guint64& id_to_remove : ids_to_remove) {
             tracked_info_map.erase(id_to_remove);
        }
        g_mutex_unlock(&filter->config_mutex); // Unlock after modification

    } // End of frame loop

    // Release meta lock
    nvds_release_meta_lock(batch_meta);

    return GST_FLOW_OK;
}

/* --- Plugin Init --- */
static gboolean
nvdszonefilter_plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(gst_nvdszonefilter_debug, "nvdszonefilter", 0, "nvdszonefilter plugin");

    return gst_element_register(plugin, "nvdszonefilter", GST_RANK_NONE, GST_TYPE_NVDSZONEFILTER);
}

/* --- Plugin Definition --- */
GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdszonefilter,
    "NvDsZoneFilter Plugin (Entry only once with debouncing and zone drawing)",
    nvdszonefilter_plugin_init,
    VERSION,
    LICENSE,
    BINARY_PACKAGE,
    URL
)