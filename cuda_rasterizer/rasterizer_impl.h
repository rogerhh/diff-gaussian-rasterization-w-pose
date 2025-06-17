/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include "forward.h"
#include <cuda_runtime_api.h>
#include <cstdint>
#include <tuple>

namespace CudaRasterizer
{
    template <typename T>
    static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
    {
        std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
        ptr = reinterpret_cast<T*>(offset);
        chunk = reinterpret_cast<char*>(ptr + count);
    }

    struct GeometryState
    {
        size_t scan_size;
        float* depths;
        char* scanning_space;
        bool* clamped;
        int* internal_radii;
        float2* means2D;
        float* cov3D;
        float4* conic_opacity;
        float* rgb;
        uint32_t* point_offsets;
        uint32_t* tiles_touched;

        static GeometryState fromChunk(char*& chunk, size_t P);
    };

    struct ImageState
    {
        uint2* ranges;
        uint32_t* n_contrib;
        float* accum_alpha;

        static ImageState fromChunk(char*& chunk, size_t N);
    };

    struct BinningState
    {
        size_t sorting_size;
        uint64_t* point_list_keys_unsorted;
        uint64_t* point_list_keys;
        uint32_t* point_list_unsorted;
        uint32_t* point_list;
        char* list_sorting_space;

        static BinningState fromChunk(char*& chunk, size_t P);
    };

    template<typename T> 
    size_t required(size_t P)
    {
        char* size = nullptr;
        T::fromChunk(size, P);
        return ((size_t)size) + 128;
    }

};  // namespace CudaRasterizer

namespace CudaRasterizer
{
        
// Forward rendering procedure for differentiable rasterization
// of Gaussians.
template <typename... JvpArgs>
int CudaRasterizer::Rasterizer::forwardJvp(
    std::function<char* (size_t)> geometryBuffer,
    std::function<char* (size_t)> binningBuffer,
    std::function<char* (size_t)> imageBuffer,
    const int P, int D, int M,
    const float* background,
    const int width, int height,
    const float* means3D,
    const float* shs,
    const float* colors_precomp,
    const float* opacities,
    const float* scales,
    const float scale_modifier,
    const float* rotations,
    const float* cov3D_precomp,
    const float tan_fovx, float tan_fovy,
    const bool prefiltered,
    float* out_color,
    float* out_depth,
    float* out_opacity,
    int* radii,
    int* n_touched,
    bool debug,
    JvpArgs&&... jvp_args)
        // const float* viewmatrix,
        // const float* projmatrix,
        // const float* cam_pos,
{
    auto jvp_args_tuple = std::forward_as_tuple(std::forward<JvpArgs>(jvp_args)...);
    auto viewmatrix = std::get<0>(jvp_args_tuple);
    auto projmatrix = std::get<1>(jvp_args_tuple);
    auto cam_pos = std::get<2>(jvp_args_tuple);

    auto focal_y = height / (2.0f * tan_fovy);
    auto focal_x = width / (2.0f * tan_fovx);

    size_t chunk_size = required<GeometryState>(P);
    char* chunkptr = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);
    

    if (radii == nullptr)
    {
        radii = geomState.internal_radii;
    }

    const int NUM_CHANNELS = 3; // Default to RGB
    const int BLOCK_X = 16;
    const int BLOCK_Y = 16;

    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Dynamically resize image-based auxiliary buffers during training
    size_t img_chunk_size = required<ImageState>(width * height);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

    if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
    {
        throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
    }


    // Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
    CHECK_CUDA(FORWARD::preprocessJvp(
        P, D, M,
        geomState.clamped,
        radii,
        tile_grid,
        geomState.tiles_touched,
        prefiltered,
        means3D,
        (glm::vec3*)scales,
        scale_modifier,
        (glm::vec4*)rotations,
        opacities,
        shs,
        cov3D_precomp,
        colors_precomp,
        viewmatrix, projmatrix,
        (glm::vec3*)cam_pos,
        width, height,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        geomState.means2D,
        geomState.depths,
        geomState.cov3D,
        geomState.rgb,
        geomState.conic_opacity
    ), debug);

    return 0;

    // // Compute prefix sum over full list of touched tile counts by Gaussians
    // // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    // CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

    // // Retrieve total number of Gaussian instances to launch and resize aux buffers
    // int num_rendered;
    // CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    // size_t binning_chunk_size = required<BinningState>(num_rendered);
    // char* binning_chunkptr = binningBuffer(binning_chunk_size);
    // BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    // // For each instance to be rendered, produce adequate [ tile | depth ] key 
    // // and corresponding dublicated Gaussian indices to be sorted
    // duplicateWithKeys << <(P + 255) / 256, 256 >> > (
    //     P,
    //     geomState.means2D,
    //     geomState.depths,
    //     geomState.point_offsets,
    //     binningState.point_list_keys_unsorted,
    //     binningState.point_list_unsorted,
    //     radii,
    //     tile_grid)
    // CHECK_CUDA(, debug)

    // int bit = getHigherMsb(tile_grid.x * tile_grid.y);

    // // Sort complete list of (duplicated) Gaussian indices by keys
    // CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
    //     binningState.list_sorting_space,
    //     binningState.sorting_size,
    //     binningState.point_list_keys_unsorted, binningState.point_list_keys,
    //     binningState.point_list_unsorted, binningState.point_list,
    //     num_rendered, 0, 32 + bit), debug)

    // CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

    // // Identify start and end of per-tile workloads in sorted list
    // if (num_rendered > 0)
    //     identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
    //         num_rendered,
    //         binningState.point_list_keys,
    //         imgState.ranges);
    // CHECK_CUDA(, debug)

    // // Let each tile blend its range of Gaussians independently in parallel
    // const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
    // CHECK_CUDA(FORWARD::render(
    //     tile_grid, block,
    //     imgState.ranges,
    //     binningState.point_list,
    //     width, height,
    //     geomState.means2D,
    //     feature_ptr,
    //     geomState.conic_opacity,
    //     imgState.accum_alpha,
    //     imgState.n_contrib,
    //     background,
    //     out_color,
    //     geomState.depths,
    //     out_depth, 
    //     out_opacity,
    //     n_touched
    // ), debug)

    // return num_rendered;
}

};  // namespace CudaRasterizer
