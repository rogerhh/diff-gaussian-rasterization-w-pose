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
#include "float_grad.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <fstream>
#include <iomanip>

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

    struct GeometryStateJvp
    {
        size_t scan_size;
        FloatGradArray<float> depths;
        char* scanning_space;
        bool* clamped;
        int* internal_radii;
        FloatGradArray<float2> means2D;
        FloatGradArray<float> cov3D;
        FloatGradArray<float4> conic_opacity;
        FloatGradArray<float> rgb;
        uint32_t* point_offsets;
        uint32_t* tiles_touched;

        static GeometryStateJvp fromChunk(char*& chunk, size_t P);
    };

    struct ImageState
    {
        uint2* ranges;
        uint32_t* n_contrib;
        float* accum_alpha;

        static ImageState fromChunk(char*& chunk, size_t N);
    };

    struct ImageStateJvp
    {
        uint2* ranges;
        uint32_t* n_contrib;
        FloatGradArray<float> accum_alpha;

        static ImageStateJvp fromChunk(char*& chunk, size_t N);
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

// Helper function to find the next-highest bit of the MSB
// on the CPU.
static inline uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}



// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ 
inline void duplicateWithKeysJvp(
	int P,
	const FloatGradArray<float2> points_xy,
	const FloatGradArray<float> depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(get_data(points_xy[idx]), radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths.data_ptr()[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ 
inline void identifyTileRangesJvp(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}
        
// Forward rendering procedure for differentiable rasterization
// of Gaussians.
template <typename... JvpArgs>
int Rasterizer::forwardJvp(JvpArgs&&... jvp_args)
    // std::function<char* (size_t)> geometryBuffer,
    // std::function<char* (size_t)> binningBuffer,
    // std::function<char* (size_t)> imageBuffer,
    // const int P, int D, int M,
    // const float* background,
    // const int width, int height,
    // const float* means3D,
    // const float* shs,
    // const float* colors_precomp,
    // const float* opacities,
    // const float* scales,
    // const float scale_modifier,
    // const float* rotations,
    // const float* cov3D_precomp,
    // const float* viewmatrix,
    // const float* projmatrix,
    // const float* cam_pos,
    // const float tan_fovx, float tan_fovy,
    // const bool prefiltered,
    // float* out_color,
    // float* out_depth,
    // float* out_opacity,
    // int* radii = nullptr,
    // int* n_touched = nullptr,
    // bool debug = false);
{
    auto jvp_args_tuple = std::forward_as_tuple(std::forward<JvpArgs>(jvp_args)...);
    std::function<char* (size_t)> geometryBuffer = std::get<0>(jvp_args_tuple);
    std::function<char* (size_t)> binningBuffer = std::get<1>(jvp_args_tuple);
    std::function<char* (size_t)> imageBuffer = std::get<2>(jvp_args_tuple);
    int P = std::get<3>(jvp_args_tuple);
    int D = std::get<4>(jvp_args_tuple);
    int M = std::get<5>(jvp_args_tuple);
    auto background = std::get<6>(jvp_args_tuple);
    int width = std::get<7>(jvp_args_tuple);
    int height = std::get<8>(jvp_args_tuple);
    auto means3D = std::get<9>(jvp_args_tuple);
    auto shs = std::get<10>(jvp_args_tuple);
    auto colors_precomp = std::get<11>(jvp_args_tuple);
    auto opacities = std::get<12>(jvp_args_tuple);
    auto scales = std::get<13>(jvp_args_tuple);
    auto scale_modifier = std::get<14>(jvp_args_tuple);
    auto rotations = std::get<15>(jvp_args_tuple);
    auto cov3D_precomp = std::get<16>(jvp_args_tuple);
    auto viewmatrix = std::get<17>(jvp_args_tuple);
    auto projmatrix = std::get<18>(jvp_args_tuple);
    auto cam_pos = std::get<19>(jvp_args_tuple);
    auto tan_fovx = std::get<20>(jvp_args_tuple);
    auto tan_fovy = std::get<21>(jvp_args_tuple);
    const bool prefiltered = std::get<22>(jvp_args_tuple);
    auto out_color = std::get<23>(jvp_args_tuple);
    auto out_depth = std::get<24>(jvp_args_tuple);
    auto out_opacity = std::get<25>(jvp_args_tuple);
    int* radii = std::get<26>(jvp_args_tuple);
    int* n_touched = std::get<27>(jvp_args_tuple);
    bool debug = std::get<28>(jvp_args_tuple);

    auto focal_y = height / (2.0f * tan_fovy);
    auto focal_x = width / (2.0f * tan_fovx);

    size_t chunk_size = required<GeometryStateJvp>(P);
    char* chunkptr = geometryBuffer(chunk_size);
    GeometryStateJvp geomState = GeometryStateJvp::fromChunk(chunkptr, P);

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
    size_t img_chunk_size = required<ImageStateJvp>(width * height);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    ImageStateJvp imgState = ImageStateJvp::fromChunk(img_chunkptr, width * height);

    if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
    {
        throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
    }

    // DEBUG reset conic opacity
    cudaMemset(geomState.depths.data_ptr(), 0, P * sizeof(float));
    cudaMemset(geomState.depths.grad_ptr(), 0, P * sizeof(float));
    cudaMemset(geomState.conic_opacity.data_ptr(), 0, P * sizeof(float4));
    cudaMemset(geomState.conic_opacity.grad_ptr(), 0, P * sizeof(float4));

    FORWARD::preprocessJvp(
        P, D, M,
        means3D,
        cast<glm::vec3>(scales),
        scale_modifier,
        cast<glm::vec4>(rotations),
        opacities,
        shs,
        geomState.clamped,
        cov3D_precomp,
        colors_precomp,
        viewmatrix, 
        projmatrix,
        cast<glm::vec3>(cam_pos),
        width, height,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        radii,
        geomState.means2D,
        geomState.depths,
        geomState.cov3D,
        geomState.rgb,
        geomState.conic_opacity,
        tile_grid,
        geomState.tiles_touched,
        prefiltered);

    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

    // Retrieve total number of Gaussian instances to launch and resize aux buffers
    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    // For each instance to be rendered, produce adequate [ tile | depth ] key 
    // and corresponding dublicated Gaussian indices to be sorted
    duplicateWithKeysJvp << <(P + 255) / 256, 256 >> > (
        P,
        geomState.means2D,
        geomState.depths,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        radii,
        tile_grid)
    CHECK_CUDA(, debug)

    int bit = getHigherMsb(tile_grid.x * tile_grid.y);

    // Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        binningState.list_sorting_space,
        binningState.sorting_size,
        binningState.point_list_keys_unsorted, binningState.point_list_keys,
        binningState.point_list_unsorted, binningState.point_list,
        num_rendered, 0, 32 + bit), debug)

    CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

    // Identify start and end of per-tile workloads in sorted list
    if (num_rendered > 0)
        identifyTileRangesJvp << <(num_rendered + 255) / 256, 256 >> > (
            num_rendered,
            binningState.point_list_keys,
            imgState.ranges);
    CHECK_CUDA(, debug)

    static_assert(is_float_grad<decltype(colors_precomp)>::value == 
                  is_float_grad<decltype(geomState.rgb)>::value,
                  "Colors precomputed and RGB must be of the same type (float or FloatGradArray).");

    // Let each tile blend its range of Gaussians independently in parallel
    auto feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
    CHECK_CUDA(FORWARD::renderJvp(
        tile_grid, block,
        imgState.ranges,
        binningState.point_list,
        width, height,
        geomState.means2D,
        feature_ptr,
        geomState.conic_opacity,
        imgState.accum_alpha,
        imgState.n_contrib,
        background,
        out_color,
        geomState.depths,
        out_depth, 
        out_opacity,
        n_touched
    ), debug)

    return num_rendered;
}


};  // namespace CudaRasterizer
