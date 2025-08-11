# SPDX-FileCopyrightText: Copyright (c) 2024 Your Name/Company
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.18) # 3.18 is generally fine for C++17
project(nvdskafka LANGUAGES C CXX)

# ---------- Options ----------------------------------------------------------
option(INSTALL_GST_PLUGIN "Install the plugin to GStreamer's plugin directory" ON)
# Force specific compilers and paths
set(CMAKE_C_COMPILER /usr/bin/gcc-10)
set(CMAKE_CXX_COMPILER /usr/bin/g++-10)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-10)
set(DS_ROOT "/opt/nvidia/deepstream/deepstream-7.1" CACHE PATH "Path to DeepStream SDK")

if(NOT EXISTS "${DS_ROOT}")
    message(FATAL_ERROR "DeepStream SDK root directory not found at ${DS_ROOT}. Please set DS_ROOT correctly.")
endif()

# --- Include Directories ---
set(DS_INCLUDE_DIRS
    "${DS_ROOT}/sources/includes"
    # Add other DS include paths if headers are in subdirectories, usually not needed for main includes
)

# --- Libraries ---
# Specify the DeepStream libraries needed by nvdskafka
set(DS_LIBRARIES
    "${DS_ROOT}/lib/libnvdsgst_meta.so" # For GstNvDsMeta, nvds_acquire_meta_lock, etc.
    "${DS_ROOT}/lib/libnvds_meta.so"    # For NvDsBatchMeta, NvDsUserMeta, nvds_get_user_meta_type, etc.
    # libnvbufsurface, libnvbufsurftransform are not needed for nvdskafka as it doesn't process buffers directly
    # Add others if your specific plugin logic requires them
)

# ---------- GStreamer --------------------------------------------------------
find_package(PkgConfig REQUIRED)
# Find required GStreamer packages
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-base-1.0)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

if(NOT GST_FOUND)
    message(FATAL_ERROR "GStreamer development files not found. Please install libgstreamer1.0-dev and libgstreamer-plugins-base1.0-dev")
endif()

# ---------- librdkafka (Kafka Client Library) --------------------------------
# Find the rdkafka library. This requires librdkafka-dev to be installed.
find_library(RDKAFKA_LIBRARY
    NAMES rdkafka
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib # Common library paths
    # NO_DEFAULT_PATH # Uncomment if you want to restrict search to PATHS only
)

find_library(UUID_LIBRARY uuid)

if(NOT UUID_LIBRARY)
    message(FATAL_ERROR "libuuid not found. Please install uuid-dev package.")
endif()

find_library(JSONCPP_LIBRARY
     NAMES jsoncpp
     PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
     NO_DEFAULT_PATH
)


find_path(RDKAFKA_INCLUDE_DIR
    NAMES librdkafka/rdkafka.h # Header might be in a subdirectory
    PATHS /usr/include /usr/local/include
    # NO_DEFAULT_PATH
)

# --- Check if librdkafka was found ---
if(NOT RDKAFKA_LIBRARY)
    message(FATAL_ERROR "librdkafka library (rdkafka) not found. Please install librdkafka-dev package.")
endif()
if(NOT RDKAFKA_INCLUDE_DIR)
    message(FATAL_ERROR "librdkafka header directory not found. Please install librdkafka-dev package.")
endif()

message(STATUS "Found librdkafka library: ${RDKAFKA_LIBRARY}")
message(STATUS "Found librdkafka headers: ${RDKAFKA_INCLUDE_DIR}")

# ---------- Compiler Settings ------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Position Independent Code is usually required for shared libraries/plugins
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------- Sources ----------------------------------------------------------
set(SRCS
    gstnvdskafka.cpp
    # Add other .cpp files here if the plugin code is split
)

# --- Create the shared library ---
add_library(nvdskafka SHARED ${SRCS})

# --- Set library properties ---
set_target_properties(nvdskafka PROPERTIES
    OUTPUT_NAME gstnvdskafka # Crucial: GStreamer expects libgst<name>.so
    PREFIX ""                 # Remove 'lib' prefix -> libgstnvdskafka.so
    INSTALL_RPATH "${DS_ROOT}/lib" # Runtime linker path for DS libs
)

# ---------- Include Directories ----------------------------------------------
target_include_directories(nvdskafka PRIVATE
    ${DS_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
    ${RDKAFKA_INCLUDE_DIR}    # Include path for librdkafka headers
    ${CMAKE_CURRENT_SOURCE_DIR} # For gstnvdskafka.h
)

# ---------- Link Libraries ---------------------------------------------------
target_link_libraries(nvdskafka PRIVATE
    ${GST_LIBRARIES}
    ${DS_LIBRARIES}
    ${RDKAFKA_LIBRARY}  
    ${UUID_LIBRARY}
    ${JSONCPP_LIBRARIES}      # Link against librdkafka
    dl                        # For dlopen, dlsym (used in the plugin)
    # pthread                 # Usually pulled in by others, but add if needed explicitly
)

# ---------- Installation (Optional) ------------------------------------------
if(INSTALL_GST_PLUGIN)
    # --- Find GStreamer plugin path ---
    find_program(GST_INSPECT gst-inspect-1.0)
    if(GST_INSPECT)
        execute_process(
            COMMAND pkg-config --variable pluginsdir gstreamer-1.0
            OUTPUT_VARIABLE GST_PLUGIN_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE PKG_CONFIG_RESULT
        )
        if(PKG_CONFIG_RESULT EQUAL 0 AND GST_PLUGIN_DIR)
            message(STATUS "GStreamer plugin directory found: ${GST_PLUGIN_DIR}")
        else()
            # Fallback path (common on Ubuntu/Debian x86_64)
            set(GST_PLUGIN_DIR "/usr/lib/x86_64-linux-gnu/gstreamer-1.0")
            message(WARNING "Could not determine GStreamer plugin directory via pkg-config, using fallback: ${GST_PLUGIN_DIR}")
        endif()
    else()
        # Fallback if gst-inspect not found
        set(GST_PLUGIN_DIR "/usr/lib/x86_64-linux-gnu/gstreamer-1.0")
        message(WARNING "gst-inspect-1.0 not found, using fallback plugin directory: ${GST_PLUGIN_DIR}")
    endif()

    # --- Install target ---
    install(TARGETS nvdskafka
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
message(STATUS "librdkafka Library: ${RDKAFKA_LIBRARY}")
message(STATUS "librdkafka Include Dir: ${RDKAFKA_INCLUDE_DIR}")