# Force specific compilers and paths
set(CMAKE_C_COMPILER /usr/bin/gcc-10)
set(CMAKE_CXX_COMPILER /usr/bin/g++-10)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-10)
set(DS_ROOT "/opt/nvidia/deepstream/deepstream-7.1" CACHE PATH "Path to DeepStream SDK")

cmake_minimum_required(VERSION 3.18)
project(instrusion_detector LANGUAGES CXX)

# Find required packages
find_package(PkgConfig REQUIRED)

# Optional: OpenCV (only if you plan to add optical flow, etc. later)
# Remove if not used
find_package(OpenCV REQUIRED COMPONENTS core imgproc cudaoptflow cudaimgproc)

# Find GStreamer packages
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_BASE REQUIRED gstreamer-base-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

set(DS_LIBRARIES
    "${DS_ROOT}/lib/libnvdsgst_meta.so"
    "${DS_ROOT}/lib/libnvds_meta.so"
    "${DS_ROOT}/lib/libnvds_msgconv.so"  # Contains schema definitions and conversion logic (might pull in schema)
    "${DS_ROOT}/lib/libnvds_utils.so"    # For nvds_get_user_meta_type, YAML parsing helpers, etc.

    # Add others if needed by your code
)

# Find DeepStream (adjust path as needed)
set(DEEPSTREAM_DIR "/opt/nvidia/deepstream/deepstream-7.1")
find_path(DEEPSTREAM_INCLUDE_DIR
    NAMES nvds_meta.h gstnvdsmeta.h
    PATHS ${DEEPSTREAM_DIR}/sources/includes
    REQUIRED
)

find_library(NVDS_META_LIBRARY
    NAMES nvdsgst_meta nvds_meta
    PATHS ${DEEPSTREAM_DIR}/lib
    REQUIRED
)

find_library(JSONCPP_LIBRARY
     NAMES jsoncpp
     PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
     NO_DEFAULT_PATH
)

find_library(NVDS_UTILS_LIBRARY
    NAMES nvds_utils
    PATHS ${DEEPSTREAM_DIR}/lib
    REQUIRED
)

# CUDA (for cudaGetDevice, etc.)
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_BASE_INCLUDE_DIRS}
    ${GSTREAMER_VIDEO_INCLUDE_DIRS}
    ${DEEPSTREAM_INCLUDE_DIR}
    /usr/local/cuda/include  # fallback
)

# Link directories
link_directories(
    ${GSTREAMER_LIBRARY_DIRS}
    ${GSTREAMER_BASE_LIBRARY_DIRS}
    ${GSTREAMER_VIDEO_LIBRARY_DIRS}
    ${DEEPSTREAM_DIR}/lib
    /usr/local/cuda/lib64
)

# Add executable
add_executable(instrusion_detector instrusion_detector.cpp)

# Link libraries
target_link_libraries(instrusion_detector PRIVATE
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_BASE_LIBRARIES}
    ${GSTREAMER_VIDEO_LIBRARIES}
    ${NVDS_META_LIBRARY}
    ${NVDS_UTILS_LIBRARY}
    ${CUDA_LIBRARIES}
    ${GST_LIBRARIES}
    ${DS_LIBRARIES}
    ${JSONCPP_LIBRARIES}
    cuda
    cudart
    pthread
    dl
    m
    rt
)

# Compiler flags
target_compile_options(instrusion_detector PRIVATE
    -Wall
    -Wno-unused-function
    -Wno-unused-variable
    -DPLATFORM_64
    -D_GLIBCXX_USE_CXX11_ABI=0
)

# Definitions
target_compile_definitions(instrusion_detector PRIVATE
    WITH_GSTREAMER
)