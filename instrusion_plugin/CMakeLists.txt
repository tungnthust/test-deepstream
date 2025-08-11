# SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Force specific compilers and paths (if needed, adjust paths/compiler versions)
# set(CMAKE_C_COMPILER /usr/bin/gcc-10)
# set(CMAKE_CXX_COMPILER /usr/bin/g++-10)
# set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-10) # Only if you have .cu files

cmake_minimum_required(VERSION 3.18) # 3.18 is generally fine for C++17/CUDA
project(nvdszonefilter LANGUAGES C CXX) # Remove CUDA if not using .cu files directly in this lib

# ---------- Options ----------------------------------------------------------
option(INSTALL_GST_PLUGIN "Install the plugin to GStreamer's plugin directory" ON)

# ---------- DeepStream -------------------------------------------------------
# --- Find DeepStream SDK ---
# Adjust DS_ROOT if your DeepStream is installed elsewhere
set(DS_ROOT "/opt/nvidia/deepstream/deepstream-7.1" CACHE PATH "Path to DeepStream SDK")

# --- Check if DS_ROOT is valid ---
if(NOT EXISTS "${DS_ROOT}")
    message(FATAL_ERROR "DeepStream SDK root directory not found at ${DS_ROOT}. Please set DS_ROOT correctly.")
endif()

# --- Include Directories ---
set(DS_INCLUDE_DIRS
    "${DS_ROOT}/sources/includes"
    # Add other DS include paths if needed by your headers
)

# --- Libraries ---
# These are the core DeepStream metadata and buffer libraries needed
# Using full paths as find_library can be tricky with .so symlinks sometimes
set(DS_LIBRARIES
    "${DS_ROOT}/lib/libnvdsgst_meta.so"
    "${DS_ROOT}/lib/libnvds_meta.so"
    # Add others if needed by your code (e.g., libnvdsgst_helper.so if used)
)

# ---------- GStreamer --------------------------------------------------------
find_package(PkgConfig REQUIRED)
# Find GStreamer packages needed
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)

# --- Check if GStreamer was found ---
if(NOT GST_FOUND)
    message(FATAL_ERROR "GStreamer development files not found. Please install libgstreamer1.0-dev and libgstreamer-plugins-base1.0-dev")
endif()

# ---------- Compiler Settings ------------------------------------------------
# --- C++ Standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Optional: CUDA (only if you add CUDA kernels like flownorm) ---
# set(CMAKE_CUDA_ARCHITECTURES 80) # A100, A30 etc. Adjust as needed.
# set(CMAKE_CUDA_STANDARD 17)

# ---------- Sources ----------------------------------------------------------
set(SRCS
    gstnvdszonefilter.cpp
    nvdszonefilter_property_parser.cpp
    # Add other .cpp files if you split the code further
)

# --- Create the shared library ---
add_library(nvdszonefilter SHARED ${SRCS})

# --- Set library properties ---
set_target_properties(nvdszonefilter PROPERTIES
    OUTPUT_NAME gstnvdszonefilter # Crucial: GStreamer expects libgst<name>.so
    PREFIX ""                 # Remove 'lib' prefix -> libgstnvdszonefilter.so
    INSTALL_RPATH "${DS_ROOT}/lib" # Runtime linker path for DS libs
)

# ---------- Include Directories ----------------------------------------------
target_include_directories(nvdszonefilter PRIVATE
    ${DS_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR} # For gstnvdszonefilter.h
    # Add other include dirs if needed
)

# ---------- Link Libraries ---------------------------------------------------
target_link_libraries(nvdszonefilter PRIVATE
    ${GST_LIBRARIES}
    ${DS_LIBRARIES}
    # Add other libraries if needed (e.g., GLib for INI parsing is usually pulled in by GStreamer)
    # If you use CUDA later, add CUDA libs like 'cuda'
)

# ---------- Installation (Optional) ------------------------------------------
if(INSTALL_GST_PLUGIN)
    # --- Find GStreamer plugin path ---
    find_program(GST_INSPECT gst-inspect-1.0)
    if(GST_INSPECT)
        # Use pkg-config to find the plugin directory
        execute_process(
            COMMAND pkg-config --variable pluginsdir gstreamer-1.0
            OUTPUT_VARIABLE GST_PLUGIN_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE PKG_CONFIG_RESULT
        )
        if(PKG_CONFIG_RESULT EQUAL 0 AND GST_PLUGIN_DIR)
            message(STATUS "GStreamer plugin directory found: ${GST_PLUGIN_DIR}")
        else()
            # Fallback path
            set(GST_PLUGIN_DIR "/usr/lib/x86_64-linux-gnu/gstreamer-1.0")
            message(WARNING "Could not determine GStreamer plugin directory via pkg-config, using fallback: ${GST_PLUGIN_DIR}")
        endif()
    else()
        # Fallback if gst-inspect not found
        set(GST_PLUGIN_DIR "/usr/lib/x86_64-linux-gnu/gstreamer-1.0")
        message(WARNING "gst-inspect-1.0 not found, using fallback plugin directory: ${GST_PLUGIN_DIR}")
    endif()

    # --- Install target ---
    install(TARGETS nvdszonefilter
            LIBRARY DESTINATION ${GST_PLUGIN_DIR}
            COMPONENT gstplugin
    )
    message(STATUS "Will install plugin to: ${GST_PLUGIN_DIR}")
endif()

# --- Debug info ---
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "DeepStream Includes: ${DS_INCLUDE_DIRS}")
message(STATUS "DeepStream Libraries: ${DS_LIBRARIES}")
message(STATUS "GStreamer Includes: ${GST_INCLUDE_DIRS}")
message(STATUS "GStreamer Libraries: ${GST_LIBRARIES}")
