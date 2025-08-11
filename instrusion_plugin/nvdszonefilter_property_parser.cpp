/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "nvdszonefilter_property_parser.h"
#include <glib.h>
#include <string.h>
#include <stdlib.h> // For atof

GST_DEBUG_CATEGORY_EXTERN(gst_nvdszonefilter_debug);
#define GST_CAT_DEFAULT gst_nvdszonefilter_debug

/**
 * @brief  Parse a zone string "x1,y1,x2,y2" into coordinates.
 * @param  zone_str  The zone string.
 * @param  x1        Pointer to store x1.
 * @param  y1        Pointer to store y1.
 * @param  x2        Pointer to store x2.
 * @param  y2        Pointer to store y2.
 * @return TRUE if parsing is successful.
 */
static gboolean
parse_zone_string(const gchar *zone_str, gdouble *x1, gdouble *y1, gdouble *x2, gdouble *y2)
{
    if (!zone_str) return FALSE;

    gchar **coords = g_strsplit(zone_str, ",", 4);
    if (!coords || g_strv_length(coords) != 4) {
        GST_ERROR("Zone string '%s' is not in the format x1,y1,x2,y2", zone_str);
        if (coords) g_strfreev(coords);
        return FALSE;
    }

    *x1 = g_ascii_strtod(coords[0], NULL);
    *y1 = g_ascii_strtod(coords[1], NULL);
    *x2 = g_ascii_strtod(coords[2], NULL);
    *y2 = g_ascii_strtod(coords[3], NULL);

    g_strfreev(coords);

    // Basic validation
    if (*x1 < 0.0 || *x1 > 1.0 || *y1 < 0.0 || *y1 > 1.0 ||
        *x2 < 0.0 || *x2 > 1.0 || *y2 < 0.0 || *y2 > 1.0) {
        GST_ERROR("Zone coordinates must be between 0.0 and 1.0. Got: %f,%f,%f,%f", *x1, *y1, *x2, *y2);
        return FALSE;
    }
    if (*x1 >= *x2 || *y1 >= *y2) {
        GST_ERROR("Zone coordinates invalid: x1 must be < x2 and y1 must be < y2. Got: %f,%f,%f,%f", *x1, *y1, *x2, *y2);
        return FALSE;
    }

    return TRUE;
}


/**
 * @brief  Callback function for GKeyFile parser.
 * @param  key_file  The GKeyFile object.
 * @param  group_name  The current group name (e.g., "stream0").
 * @param  keys  The list of keys in the group.
 * @param  user_data  Pointer to GstNvDsZoneFilter instance.
 */
static void
parse_group(GKeyFile *key_file, const gchar *group_name, gchar **keys, gpointer user_data)
{
    GstNvDsZoneFilter *filter = GST_NVDSZONEFILTER(user_data);

    if (g_strcmp0(group_name, "global") == 0) {
        // We could parse global settings here if needed, but for now, we don't.
        return;
    }

    // Check if the group name starts with "stream"
    if (g_str_has_prefix(group_name, "stream")) {
        gchar *endptr = NULL;
        gint64 stream_id = g_ascii_strtoll(group_name + 6, &endptr, 10); // "stream" + 6 chars = start of number

        if (*endptr != '\0' || stream_id < 0) {
            GST_WARNING("Invalid stream group name format: %s", group_name);
            return;
        }

        GError *error = NULL;
        gchar *zone_str = g_key_file_get_string(key_file, group_name, "zone", &error);

        if (error) {
            GST_WARNING("Failed to read 'zone' for group %s: %s", group_name, error->message);
            g_error_free(error);
            return;
        }

        StreamZoneConfig zone_config = {0};
        zone_config.stream_id = (gint)stream_id;
        zone_config.valid = FALSE; // Default to invalid

        if (parse_zone_string(zone_str, &zone_config.x1, &zone_config.y1, &zone_config.x2, &zone_config.y2)) {
            zone_config.valid = TRUE;
            GST_INFO_OBJECT(filter, "Parsed zone for stream %d: (%.2f,%.2f) to (%.2f,%.2f)",
                            zone_config.stream_id, zone_config.x1, zone_config.y1, zone_config.x2, zone_config.y2);
        } else {
            GST_ERROR_OBJECT(filter, "Failed to parse zone string '%s' for stream %d", zone_str, zone_config.stream_id);
            // We still store the invalid config to indicate we tried
        }

        // Store the configuration in the map
        (*filter->stream_zones_map)[zone_config.stream_id] = zone_config;
        g_free(zone_str);
    }
}


/**
 * @brief  Main function to parse the configuration file.
 * @param  filter  Pointer to the GstNvDsZoneFilter instance.
 * @param  cfg_file_path  Path to the configuration file.
 * @return TRUE if parsing is successful.
 */
gboolean
nvdszonefilter_parse_config_file(GstNvDsZoneFilter *filter, const gchar *cfg_file_path)
{
    GKeyFile *key_file;
    GError *error = NULL;
    gboolean ret = FALSE;
    printf("Configuration file parsed: %s\n", cfg_file_path);
    // --- Move variable declarations here ---
    gchar **groups = NULL; 
    // --- End of moved declarations ---

    key_file = g_key_file_new();

    if (!g_key_file_load_from_file(key_file, cfg_file_path, G_KEY_FILE_NONE, &error)) {
        GST_ELEMENT_ERROR(filter, RESOURCE, NOT_FOUND, ("Failed to load config file"),
                          ("Config file path: %s, Error: %s", cfg_file_path, error->message));
        g_error_free(error);
        goto done; // This goto is now safe as 'groups' is declared above
    }

    // Clear any previous configuration
    filter->stream_zones_map->clear();

    // --- Get groups after potential goto ---
    groups = g_key_file_get_groups(key_file, NULL);
    // --- End of moved initialization ---
    
    if (groups) {
        for (int i = 0; groups[i] != NULL; i++) {
            gchar **keys = g_key_file_get_keys(key_file, groups[i], NULL, NULL); // This is fine inside the block
            if (keys) {
                parse_group(key_file, groups[i], keys, filter);
                g_strfreev(keys);
            }
        }
        g_strfreev(groups); // Free the groups array
        groups = NULL; // Good practice to NULL after free if variable persists
    }

    ret = TRUE; // If we got here, basic parsing was successful
    GST_INFO_OBJECT(filter, "Configuration file parsed successfully: %s", cfg_file_path);

done:
    // Ensure resources are freed even if an error occurred early
    if (groups) {
        g_strfreev(groups);
    }
    if (key_file) {
        g_key_file_free(key_file);
    }
    return ret;
}