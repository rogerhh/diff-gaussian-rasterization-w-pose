#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

cmake_minimum_required(VERSION 3.20)

project(DiffRast LANGUAGES CUDA CXX)

include(FetchContent)

# Fetch GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_library(CudaRasterizer
    cuda_rasterizer/backward.h
    cuda_rasterizer/backward.cu
    cuda_rasterizer/forward.h
    cuda_rasterizer/forward_impl.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/auxiliary.h
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/rasterizer.h
)

set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86")

target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_include_directories(CudaRasterizer PRIVATE third_party/glm third_party/auto-jvp-example ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(basic_tests
    tests/main.cu
    tests/test_utils.cu
    tests/transform4x4_test.cu
    tests/in_frustum_test.cu
    tests/vec4_tests.cu
)

target_include_directories(basic_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer ${CMAKE_CURRENT_SOURCE_DIR}/tests third_party/glm third_party/auto-jvp-example)
target_link_libraries(basic_tests PRIVATE CudaRasterizer gtest gtest_main)

# Copy tests/data into build directory
file(COPY tests/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

