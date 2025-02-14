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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,
                const int selected_pixel_size,
                const int* selected_pixel_indices,
                const int selected_gaussian_size,
                const int* selected_gaussian_indices,
                const bool* selected_gaussian_bools,
		const float* dL_dpixels,
		const float* dL_dpixels_depth,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_ddepths);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float* proj_raw,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
                const int selected_gaussian_size,
                const int* selected_gaussian_indices,
                const bool* selected_gaussian_bools,
		const float3* dL_dmean2D,
		const float* dL_dconics,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_ddepth,
		float* dL_dcov3D,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		float* dL_dtau);

	void renderSketchJacobian(
                const int sketch_mode,
                const int sketch_dim,
                const int* sketch_indices,
		const dim3 grid, const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
                int P,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,
                const int selected_pixel_size,
                const int* selected_pixel_indices,
                const int selected_gaussian_size,
                const int* selected_gaussian_indices,
                const bool* selected_gaussian_bools,
		const float* df_dpixels,
		const float* df_dpixels_depth,
		float3* df_dmean2D,
		float4* df_dconic2D,
		float* df_dopacity,
		float* df_dcolors,
		float* df_ddepths);

	void preprocessSketchJacobian(
                const int sketch_mode,
                const int sketch_dim,
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float* proj_raw,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
                const int selected_gaussian_size,
                const int* selected_gaussian_indices,
                const bool* selected_gaussian_bools,
		const float3* df_dmean2D,
		const float* df_dconics,
		glm::vec3* df_dmeans,
		float* df_dcolor,
		float* df_ddepth,
		float* df_dcov3D,
		float* df_dsh,
		glm::vec3* df_dscale,
		glm::vec4* df_drot,
		float* df_dtau,
                const float* df_dopacity);
}

#endif
