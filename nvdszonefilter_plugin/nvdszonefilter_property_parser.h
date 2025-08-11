/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */


#ifndef _NVDSZONEFILTER_PROPERTY_PARSER_H_
#define _NVDSZONEFILTER_PROPERTY_PARSER_H_

#include <gst/gst.h>
#include <unordered_map>
#include "gstnvdszonefilter.h" // For StreamZoneConfig

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Parse the zone configuration file.
 * @param  filter  Pointer to the GstNvDsZoneFilter instance.
 * @param  cfg_file_path  Path to the configuration file.
 * @return TRUE if parsing is successful.
 */
gboolean nvdszonefilter_parse_config_file(GstNvDsZoneFilter *filter, const gchar *cfg_file_path);

#ifdef __cplusplus
}
#endif

#endif /* _NVDSZONEFILTER_PROPERTY_PARSER_H_ */