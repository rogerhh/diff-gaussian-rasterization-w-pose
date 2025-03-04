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

#include "backward.h"
#include "auxiliary.h"
#include "math.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <iostream>
#include <cuda_runtime.h>

#define MAX_SKETCH_DIM 32

namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int global_idx, int gaussian_idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* df_dcolor, glm::vec3* df_dmeans, glm::vec3* df_dshs,  float *df_dtau)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[gaussian_idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + gaussian_idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 df_dRGB = df_dcolor[global_idx];
	df_dRGB.x *= clamped[3 * gaussian_idx + 0] ? 0 : 1;
	df_dRGB.y *= clamped[3 * gaussian_idx + 1] ? 0 : 1;
	df_dRGB.z *= clamped[3 * gaussian_idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* df_dsh = df_dshs + global_idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	df_dsh[0] = dRGBdsh0 * df_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		df_dsh[1] = dRGBdsh1 * df_dRGB;
		df_dsh[2] = dRGBdsh2 * df_dRGB;
		df_dsh[3] = dRGBdsh3 * df_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			df_dsh[4] = dRGBdsh4 * df_dRGB;
			df_dsh[5] = dRGBdsh5 * df_dRGB;
			df_dsh[6] = dRGBdsh6 * df_dRGB;
			df_dsh[7] = dRGBdsh7 * df_dRGB;
			df_dsh[8] = dRGBdsh8 * df_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				df_dsh[9] = dRGBdsh9 * df_dRGB;
				df_dsh[10] = dRGBdsh10 * df_dRGB;
				df_dsh[11] = dRGBdsh11 * df_dRGB;
				df_dsh[12] = dRGBdsh12 * df_dRGB;
				df_dsh[13] = dRGBdsh13 * df_dRGB;
				df_dsh[14] = dRGBdsh14 * df_dRGB;
				df_dsh[15] = dRGBdsh15 * df_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 df_ddir(glm::dot(dRGBdx, df_dRGB), glm::dot(dRGBdy, df_dRGB), glm::dot(dRGBdz, df_dRGB));

	// Account for normalization of direction
	float3 df_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ df_ddir.x, df_ddir.y, df_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	df_dmeans[global_idx] += glm::vec3(df_dmean.x, df_dmean.y, df_dmean.z);

	df_dtau[6 * global_idx + 0] += -df_dmean.x;
	df_dtau[6 * global_idx + 1] += -df_dmean.y;
	df_dtau[6 * global_idx + 2] += -df_dmean.z;

}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DSketchJacobianCUDA(
        const int sketch_mode,
        const int sketch_dim,
        int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
        const int selected_gaussian_size,
        const int* __restrict__ selected_gaussian_indices,
        const bool* __restrict__ selected_gaussian_bools,
	const float* df_dconics,
	float3* df_dmeans,
	float* df_dcov,
	float *df_dtau)
{
        auto block = cg::this_thread_block();   // Each block computes a pixel
        auto tid = block.thread_rank();

        int block_x = block.group_index().x;
        int block_y = block.group_index().y;
        int thread_x = block.thread_index().x;
        int thread_y = block.thread_index().y;

        auto sketch_idx = block_x * blockDim.x + thread_x;    // This is the index of the sketched row
        auto gaussian_idx = block_y * blockDim.y + thread_y;    // This is the index of the gaussian splat
        auto global_idx = gaussian_idx * sketch_dim + sketch_idx;
        // auto global_idx = sketch_idx * P + gaussian_idx;


        if (gaussian_idx >= P || sketch_idx >= sketch_dim) {
            return;
        }

	if (!(radii[gaussian_idx] > 0))
	    return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * gaussian_idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[gaussian_idx];
	float3 df_dconic = { df_dconics[4 * global_idx], df_dconics[4 * global_idx + 1], df_dconics[4 * global_idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float df_da = 0, df_db = 0, df_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., df / da = df / d_conic_a * d_conic_a / d_a
		df_da = denom2inv * (-c * c * df_dconic.x + 2 * b * c * df_dconic.y + (denom - a * c) * df_dconic.z);
		df_dc = denom2inv * (-a * a * df_dconic.z + 2 * a * b * df_dconic.y + (denom - a * c) * df_dconic.x);
		df_db = denom2inv * 2 * (b * c * df_dconic.x - (denom + 2 * b * b) * df_dconic.y + a * b * df_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		df_dcov[6 * global_idx + 0] = (T[0][0] * T[0][0] * df_da + T[0][0] * T[1][0] * df_db + T[1][0] * T[1][0] * df_dc);
		df_dcov[6 * global_idx + 3] = (T[0][1] * T[0][1] * df_da + T[0][1] * T[1][1] * df_db + T[1][1] * T[1][1] * df_dc);
		df_dcov[6 * global_idx + 5] = (T[0][2] * T[0][2] * df_da + T[0][2] * T[1][2] * df_db + T[1][2] * T[1][2] * df_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		df_dcov[6 * global_idx + 1] = 2 * T[0][0] * T[0][1] * df_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * df_db + 2 * T[1][0] * T[1][1] * df_dc;
		df_dcov[6 * global_idx + 2] = 2 * T[0][0] * T[0][2] * df_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * df_db + 2 * T[1][0] * T[1][2] * df_dc;
		df_dcov[6 * global_idx + 4] = 2 * T[0][2] * T[0][1] * df_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * df_db + 2 * T[1][1] * T[1][2] * df_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			df_dcov[6 * global_idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float df_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * df_da +
	        (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * df_db;
	float df_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * df_da +
	        (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * df_db;
	float df_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * df_da +
	        (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * df_db;
	float df_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * df_dc +
	        (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * df_db;
	float df_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * df_dc +
	        (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * df_db;
	float df_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * df_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * df_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float df_dJ00 = W[0][0] * df_dT00 + W[0][1] * df_dT01 + W[0][2] * df_dT02;
	float df_dJ02 = W[2][0] * df_dT00 + W[2][1] * df_dT01 + W[2][2] * df_dT02;
	float df_dJ11 = W[1][0] * df_dT10 + W[1][1] * df_dT11 + W[1][2] * df_dT12;
	float df_dJ12 = W[2][0] * df_dT10 + W[2][1] * df_dT11 + W[2][2] * df_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float df_dtx = x_grad_mul * -h_x * tz2 * df_dJ02;
	float df_dty = y_grad_mul * -h_y * tz2 * df_dJ12;
	float df_dtz = -h_x * tz2 * df_dJ00 - h_y * tz2 * df_dJ11 + (2 * h_x * t.x) * tz3 * df_dJ02 + (2 * h_y * t.y) * tz3 * df_dJ12;

	SE3 T_CW(view_matrix);
	mat33 R = T_CW.R().data();
	mat33 RT = R.transpose();
	float3 t_ = T_CW.t();
	mat33 dpC_drho = mat33::identity();
	mat33 dpC_dtheta = -mat33::skew_symmetric(t);
	float df_dt[6];
	for (int i = 0; i < 3; i++) {
		float3 c_rho = dpC_drho.cols[i];
		float3 c_theta = dpC_dtheta.cols[i];
		df_dt[i] = df_dtx * c_rho.x + df_dty * c_rho.y + df_dtz * c_rho.z;
		df_dt[i + 3] = df_dtx * c_theta.x + df_dty * c_theta.y + df_dtz * c_theta.z;
	}
	for (int i = 0; i < 6; i++) {
		df_dtau[6 * global_idx + i] += df_dt[i];
	}

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 df_dmean = transformVec4x3Transpose({ df_dtx, df_dty, df_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	df_dmeans[global_idx] = df_dmean;

	float df_dW00 = J[0][0] * df_dT00;
	float df_dW01 = J[0][0] * df_dT01;
	float df_dW02 = J[0][0] * df_dT02;
	float df_dW10 = J[1][1] * df_dT10;
	float df_dW11 = J[1][1] * df_dT11;
	float df_dW12 = J[1][1] * df_dT12;
	float df_dW20 = J[0][2] * df_dT00 + J[1][2] * df_dT10;
	float df_dW21 = J[0][2] * df_dT01 + J[1][2] * df_dT11;
	float df_dW22 = J[0][2] * df_dT02 + J[1][2] * df_dT12;

	float3 c1 = R.cols[0];
	float3 c2 = R.cols[1];
	float3 c3 = R.cols[2];

	float df_dW_data[9];
	df_dW_data[0] = df_dW00;
	df_dW_data[3] = df_dW01;
	df_dW_data[6] = df_dW02;
	df_dW_data[1] = df_dW10;
	df_dW_data[4] = df_dW11;
	df_dW_data[7] = df_dW12;
	df_dW_data[2] = df_dW20;
	df_dW_data[5] = df_dW21;
	df_dW_data[8] = df_dW22;

	mat33 df_dW(df_dW_data);
	float3 df_dWc1 = df_dW.cols[0];
	float3 df_dWc2 = df_dW.cols[1];
	float3 df_dWc3 = df_dW.cols[2];

	mat33 n_W1_x = -mat33::skew_symmetric(c1);
	mat33 n_W2_x = -mat33::skew_symmetric(c2);
	mat33 n_W3_x = -mat33::skew_symmetric(c3);

	float3 df_dtheta = {};
	df_dtheta.x = dot(df_dWc1, n_W1_x.cols[0]) + dot(df_dWc2, n_W2_x.cols[0]) +
				dot(df_dWc3, n_W3_x.cols[0]);
	df_dtheta.y = dot(df_dWc1, n_W1_x.cols[1]) + dot(df_dWc2, n_W2_x.cols[1]) +
				dot(df_dWc3, n_W3_x.cols[1]);
	df_dtheta.z = dot(df_dWc1, n_W1_x.cols[2]) + dot(df_dWc2, n_W2_x.cols[2]) +
				dot(df_dWc3, n_W3_x.cols[2]);

	df_dtau[6 * global_idx + 3] += df_dtheta.x;
	df_dtau[6 * global_idx + 4] += df_dtheta.y;
	df_dtau[6 * global_idx + 5] += df_dtheta.z;

}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int global_idx, int gaussian_idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* df_dcov3Ds, glm::vec3* df_dscales, glm::vec4* df_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* df_dcov3D = df_dcov3Ds + 6 * global_idx;

	glm::vec3 dunc(df_dcov3D[0], df_dcov3D[3], df_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(df_dcov3D[1], df_dcov3D[2], df_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 df_dSigma = glm::mat3(
		df_dcov3D[0], 0.5f * df_dcov3D[1], 0.5f * df_dcov3D[2],
		0.5f * df_dcov3D[1], df_dcov3D[3], 0.5f * df_dcov3D[4],
		0.5f * df_dcov3D[2], 0.5f * df_dcov3D[4], df_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 df_dM = 2.0f * M * df_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 df_dMt = glm::transpose(df_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* df_dscale = df_dscales + global_idx;
	df_dscale->x = glm::dot(Rt[0], df_dMt[0]);
	df_dscale->y = glm::dot(Rt[1], df_dMt[1]);
	df_dscale->z = glm::dot(Rt[2], df_dMt[2]);

	df_dMt[0] *= s.x;
	df_dMt[1] *= s.y;
	df_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 df_dq;
	df_dq.x = 2 * z * (df_dMt[0][1] - df_dMt[1][0]) + 2 * y * (df_dMt[2][0] - df_dMt[0][2]) + 2 * x * (df_dMt[1][2] - df_dMt[2][1]);
	df_dq.y = 2 * y * (df_dMt[1][0] + df_dMt[0][1]) + 2 * z * (df_dMt[2][0] + df_dMt[0][2]) + 2 * r * (df_dMt[1][2] - df_dMt[2][1]) - 4 * x * (df_dMt[2][2] + df_dMt[1][1]);
	df_dq.z = 2 * x * (df_dMt[1][0] + df_dMt[0][1]) + 2 * r * (df_dMt[2][0] - df_dMt[0][2]) + 2 * z * (df_dMt[1][2] + df_dMt[2][1]) - 4 * y * (df_dMt[2][2] + df_dMt[0][0]);
	df_dq.w = 2 * r * (df_dMt[0][1] - df_dMt[1][0]) + 2 * x * (df_dMt[2][0] + df_dMt[0][2]) + 2 * y * (df_dMt[1][2] + df_dMt[2][1]) - 4 * z * (df_dMt[1][1] + df_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* df_drot = (float4*)(df_drots + global_idx);
	*df_drot = float4{ df_dq.x, df_dq.y, df_dq.z, df_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ df_dq.x, df_dq.y, df_dq.z, df_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessSketchJacobianCUDA(
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
	const float *viewmatrix,
	const float* proj,
	const float *proj_raw,
	const glm::vec3* campos,
        const int selected_gaussian_size,
        const int* __restrict__ selected_gaussian_indices,
        const bool* __restrict__ selected_gaussian_bools,
	const float3* df_dmean2D,
	glm::vec3* df_dmeans,
	float* df_dcolor,
	float *df_ddepth,
	float* df_dcov3D,
	float* df_dsh,
	glm::vec3* df_dscale,
	glm::vec4* df_drot,
	float *df_dtau)
{
        auto block = cg::this_thread_block();   // Each block computes a pixel
        auto tid = block.thread_rank();

        int block_x = block.group_index().x;
        int block_y = block.group_index().y;
        int thread_x = block.thread_index().x;
        int thread_y = block.thread_index().y;

        auto sketch_idx = block_x * blockDim.x + thread_x;    // This is the index of the sketched row
        auto gaussian_idx = block_y * blockDim.y + thread_y;    // This is the index of the gaussian splat
        auto global_idx = gaussian_idx * sketch_dim + sketch_idx;
        // auto global_idx = sketch_idx * P + gaussian_idx;


        if (gaussian_idx >= P || sketch_idx >= sketch_dim) {
            return;
        }

	if (!(radii[gaussian_idx] > 0))
	    return;

	float3 m = means[gaussian_idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 df_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	df_dmean.x = (proj[0] * m_w - proj[3] * mul1) * df_dmean2D[global_idx].x + (proj[1] * m_w - proj[3] * mul2) * df_dmean2D[global_idx].y;
	df_dmean.y = (proj[4] * m_w - proj[7] * mul1) * df_dmean2D[global_idx].x + (proj[5] * m_w - proj[7] * mul2) * df_dmean2D[global_idx].y;
	df_dmean.z = (proj[8] * m_w - proj[11] * mul1) * df_dmean2D[global_idx].x + (proj[9] * m_w - proj[11] * mul2) * df_dmean2D[global_idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	df_dmeans[global_idx] += df_dmean;

	float alpha = 1.0f * m_w;
	float beta = -m_hom.x * m_w * m_w;
	float gamma = -m_hom.y * m_w * m_w;

	float a = proj_raw[0];
	float b = proj_raw[5];
	float c = proj_raw[10];
	float d = proj_raw[14];
	float e = proj_raw[11];

	SE3 T_CW(viewmatrix);
	mat33 R = T_CW.R().data();
	mat33 RT = R.transpose();
	float3 t = T_CW.t();
	float3 p_C = T_CW * m;
	mat33 dp_C_d_rho = mat33::identity();
	mat33 dp_C_d_theta = -mat33::skew_symmetric(p_C);

	float3 d_proj_dp_C1 = make_float3(alpha * a, 0.f, beta * e);
	float3 d_proj_dp_C2 = make_float3(0.f, alpha * b, gamma * e);

	float3 d_proj_dp_C1_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C1; // x.T A = A.T x
	float3 d_proj_dp_C2_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C2;
	float3 d_proj_dp_C1_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C1;
	float3 d_proj_dp_C2_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C2;

	float2 dmean2D_dtau[6];
	dmean2D_dtau[0].x = d_proj_dp_C1_d_rho.x;
	dmean2D_dtau[1].x = d_proj_dp_C1_d_rho.y;
	dmean2D_dtau[2].x = d_proj_dp_C1_d_rho.z;
	dmean2D_dtau[3].x = d_proj_dp_C1_d_theta.x;
	dmean2D_dtau[4].x = d_proj_dp_C1_d_theta.y;
	dmean2D_dtau[5].x = d_proj_dp_C1_d_theta.z;

	dmean2D_dtau[0].y = d_proj_dp_C2_d_rho.x;
	dmean2D_dtau[1].y = d_proj_dp_C2_d_rho.y;
	dmean2D_dtau[2].y = d_proj_dp_C2_d_rho.z;
	dmean2D_dtau[3].y = d_proj_dp_C2_d_theta.x;
	dmean2D_dtau[4].y = d_proj_dp_C2_d_theta.y;
	dmean2D_dtau[5].y = d_proj_dp_C2_d_theta.z;

	float df_dt[6];
	for (int i = 0; i < 6; i++) {
		df_dt[i] = df_dmean2D[global_idx].x * dmean2D_dtau[i].x + df_dmean2D[global_idx].y * dmean2D_dtau[i].y;
	}
	for (int i = 0; i < 6; i++) {
		df_dtau[6 * global_idx + i] += df_dt[i];
	}

	// Compute gradient update due to computing depths
	// p_orig = m
	// p_view = transformPoint4x3(p_orig, viewmatrix);
	// depth = p_view.z;
	float df_dpCz = df_ddepth[global_idx];
	df_dmeans[global_idx].x += df_dpCz * viewmatrix[2];
	df_dmeans[global_idx].y += df_dpCz * viewmatrix[6];
	df_dmeans[global_idx].z += df_dpCz * viewmatrix[10];

	for (int i = 0; i < 3; i++) {
		float3 c_rho = dp_C_d_rho.cols[i];
		float3 c_theta = dp_C_d_theta.cols[i];
		df_dtau[6 * global_idx + i] += df_dpCz * c_rho.z;
		df_dtau[6 * global_idx + i + 3] += df_dpCz * c_theta.z;
	}

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(global_idx, gaussian_idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)df_dcolor, (glm::vec3*)df_dmeans, (glm::vec3*)df_dsh, df_dtau);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(global_idx, gaussian_idx, scales[gaussian_idx], scale_modifier, rotations[gaussian_idx], df_dcov3D, df_dscale, df_drot);
}

template <typename T>
__device__ void inline reduce_helper(int lane, int i, T *data) {
  if (lane < i) {
    data[lane] += data[lane + i];
  }
}

template <typename group_t, typename... Lists>
__device__ void render_cuda_reduce_sum(group_t g, Lists... lists) {
  int lane = g.thread_rank();
  g.sync();

  for (int i = g.size() / 2; i > 0; i /= 2) {
    (...,
     reduce_helper(
         lane, i, lists)); // Fold expression: apply reduce_helper for each list
    g.sync();
  }
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderSketchJacobianCUDA(
        const int sketch_mode,
        const int sketch_dim,
        const int* __restrict__ sketch_indices,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
        int P,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
        const int selected_gaussian_size,
        const int* __restrict__ selected_gaussian_indices,
        const bool* __restrict__ selected_gaussian_bools,
	const float* __restrict__ df_dpixels,
	const float* __restrict__ df_dpixels_depth,
	float3* __restrict__ df_dmean2D,
	float4* __restrict__ df_dconic2D,
	float* __restrict__ df_dopacity,
	float* __restrict__ df_dcolors,
	float* __restrict__ df_ddepths)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	auto tid = block.thread_rank();
    
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

        const int sketch_idx = sketch_mode == 0? 0 : sketch_indices[pix_id];
        if (sketch_idx >= sketch_dim || sketch_idx < 0) {
            return;
        }

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];

	__shared__ float2 df_dmean2D_shared[BLOCK_SIZE];
	__shared__ float3 df_dcolors_shared[BLOCK_SIZE];
	__shared__ float df_ddepths_shared[BLOCK_SIZE];
	__shared__ float df_dopacity_shared[BLOCK_SIZE];
	__shared__ float4 df_dconic2D_shared[BLOCK_SIZE];

        __shared__ float2 df_dmean2D_sketch_shared[MAX_SKETCH_DIM];
        __shared__ float3 df_dcolors_sketch_shared[MAX_SKETCH_DIM];
        __shared__ float df_ddepths_sketch_shared[MAX_SKETCH_DIM];
        __shared__ float df_dopacity_sketch_shared[MAX_SKETCH_DIM];
        __shared__ float4 df_dconic2D_sketch_shared[MAX_SKETCH_DIM];


	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float df_dpixel[C] = { 0 };
	float accum_rec_depth = 0;
	float df_dpixel_depth = 0;
	if (inside) {
		#pragma unroll
		for (int i = 0; i < C; i++) {
			df_dpixel[i] = df_dpixels[i * H * W + pix_id];
		}
		df_dpixel_depth = df_dpixels_depth[pix_id];
	}

	float last_alpha = 0.f;
	float last_color[C] = { 0.f };
	float last_depth = 0.f;

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;
	__shared__ int skip_counter;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		// block.sync();
		const int progress = i * BLOCK_SIZE + tid;
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[tid] = coll_id;
			collected_xy[tid] = points_xy_image[coll_id];
			collected_conic_opacity[tid] = conic_opacity[coll_id];
			#pragma unroll
			for (int i = 0; i < C; i++) {
				collected_colors[i * BLOCK_SIZE + tid] = colors[coll_id * C + i];
				
			}
			collected_depths[tid] = depths[coll_id];
		}
		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) {
			block.sync();
			if (tid == 0) {
				skip_counter = 0;
			}

                        // Reset the shared memory for the sketch gradients
                        if (tid < sketch_dim) {
                            df_dmean2D_sketch_shared[tid] = make_float2(0.0f, 0.0f);
                            df_dconic2D_sketch_shared[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                            df_dopacity_sketch_shared[tid] = 0.0f;
                            df_dcolors_sketch_shared[tid] = make_float3(0.0f, 0.0f, 0.0f);
                            df_ddepths_sketch_shared[tid] = 0.0f;
                        }
			block.sync();


			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			bool skip = done;
			contributor = done ? contributor : contributor - 1;
			skip |= contributor >= last_contributor;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			skip |= power > 0.0f;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			skip |= alpha < 1.0f / 255.0f;

			if (skip) {
				atomicAdd(&skip_counter, 1);
			}
			block.sync();
			if (skip_counter == BLOCK_SIZE) {
				continue;
			}


			T = skip ? T : T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float df_dalpha = 0.0f;
			const int global_id = collected_id[j];
			float local_df_dcolors[3];
			#pragma unroll
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = skip ? accum_rec[ch] : last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = skip ? last_color[ch] : c;

				const float df_dchannel = df_dpixel[ch];
				df_dalpha += (c - accum_rec[ch]) * df_dchannel;
				local_df_dcolors[ch] = skip ? 0.0f : dchannel_dcolor * df_dchannel;
			}
			df_dcolors_shared[tid].x = local_df_dcolors[0];
			df_dcolors_shared[tid].y = local_df_dcolors[1];
			df_dcolors_shared[tid].z = local_df_dcolors[2];

			const float depth = collected_depths[j];
			accum_rec_depth = skip ? accum_rec_depth : last_alpha * last_depth + (1.f - last_alpha) * accum_rec_depth;
			last_depth = skip ? last_depth : depth;
			df_dalpha += (depth - accum_rec_depth) * df_dpixel_depth;
			df_ddepths_shared[tid] = skip ? 0.f : dchannel_dcolor * df_dpixel_depth;


			df_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = skip ? last_alpha : alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0.f;
			#pragma unroll
			for (int i = 0; i < C; i++) {
				bg_dot_dpixel +=  bg_color[i] * df_dpixel[i];
			}
			df_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// Helpful reusable temporary variables
			const float df_dG = con_o.w * df_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

                        // First sum to local shared memory
			df_dmean2D_shared[tid].x = skip ? 0.f : df_dG * dG_ddelx * ddelx_dx;
			df_dmean2D_shared[tid].y = skip ? 0.f : df_dG * dG_ddely * ddely_dy;
			df_dconic2D_shared[tid].x = skip ? 0.f : -0.5f * gdx * d.x * df_dG;
			df_dconic2D_shared[tid].y = skip ? 0.f : -0.5f * gdx * d.y * df_dG;
			df_dconic2D_shared[tid].w = skip ? 0.f : -0.5f * gdy * d.y * df_dG;
			df_dopacity_shared[tid] = skip ? 0.f : G * df_dalpha;

			// render_cuda_reduce_sum(block, 
			// 	df_dmean2D_shared,
			// 	df_dconic2D_shared,
			// 	df_dopacity_shared,
			// 	df_dcolors_shared, 
			// 	df_ddepths_shared
			// );	
			
			// if (tid == 0) {
			// 	float2 df_dmean2D_acc = df_dmean2D_shared[0];
			// 	float4 df_dconic2D_acc = df_dconic2D_shared[0];
			// 	float df_dopacity_acc = df_dopacity_shared[0];
			// 	float3 df_dcolors_acc = df_dcolors_shared[0];
			// 	float df_ddepths_acc = df_ddepths_shared[0];

			// 	atomicAdd(&df_dmean2D[global_id].x, df_dmean2D_acc.x);
			// 	atomicAdd(&df_dmean2D[global_id].y, df_dmean2D_acc.y);
			// 	atomicAdd(&df_dconic2D[global_id].x, df_dconic2D_acc.x);
			// 	atomicAdd(&df_dconic2D[global_id].y, df_dconic2D_acc.y);
			// 	atomicAdd(&df_dconic2D[global_id].w, df_dconic2D_acc.w);
			// 	atomicAdd(&df_dopacity[global_id], df_dopacity_acc);
			// 	atomicAdd(&df_dcolors[global_id * C + 0], df_dcolors_acc.x);
			// 	atomicAdd(&df_dcolors[global_id * C + 1], df_dcolors_acc.y);
			// 	atomicAdd(&df_dcolors[global_id * C + 2], df_dcolors_acc.z);
			// 	atomicAdd(&df_ddepths[global_id], df_ddepths_acc);
			// }

                        atomicAdd(&df_dmean2D_sketch_shared[sketch_idx].x, df_dmean2D_shared[tid].x);
                        atomicAdd(&df_dmean2D_sketch_shared[sketch_idx].y, df_dmean2D_shared[tid].y);
                        atomicAdd(&df_dconic2D_sketch_shared[sketch_idx].x, df_dconic2D_shared[tid].x);
                        atomicAdd(&df_dconic2D_sketch_shared[sketch_idx].y, df_dconic2D_shared[tid].y);
                        atomicAdd(&df_dconic2D_sketch_shared[sketch_idx].w, df_dconic2D_shared[tid].w);
                        atomicAdd(&df_dopacity_sketch_shared[sketch_idx], df_dopacity_shared[tid]);
                        atomicAdd(&df_dcolors_sketch_shared[sketch_idx].x, df_dcolors_shared[tid].x);
                        atomicAdd(&df_dcolors_sketch_shared[sketch_idx].y, df_dcolors_shared[tid].y);
                        atomicAdd(&df_dcolors_sketch_shared[sketch_idx].z, df_dcolors_shared[tid].z);
                        atomicAdd(&df_ddepths_sketch_shared[sketch_idx], df_ddepths_shared[tid]);

                        block.sync();

                        // Then sum to global memory
                        if (tid < sketch_dim) {
                            int dest_global_id = global_id * sketch_dim + tid;
                            // int dest_global_id = tid * P + global_id;
                            atomicAdd(&df_dmean2D[dest_global_id].x, df_dmean2D_sketch_shared[tid].x);
                            atomicAdd(&df_dmean2D[dest_global_id].y, df_dmean2D_sketch_shared[tid].y);
                            atomicAdd(&df_dconic2D[dest_global_id].x, df_dconic2D_sketch_shared[tid].x);
                            atomicAdd(&df_dconic2D[dest_global_id].y, df_dconic2D_sketch_shared[tid].y);
                            atomicAdd(&df_dconic2D[dest_global_id].w, df_dconic2D_sketch_shared[tid].w);
                            atomicAdd(&df_dopacity[dest_global_id], df_dopacity_sketch_shared[tid]);
                            atomicAdd(&df_dcolors[dest_global_id * C + 0], df_dcolors_sketch_shared[tid].x);
                            atomicAdd(&df_dcolors[dest_global_id * C + 1], df_dcolors_sketch_shared[tid].y);
                            atomicAdd(&df_dcolors[dest_global_id * C + 2], df_dcolors_sketch_shared[tid].z);
                            atomicAdd(&df_ddepths[dest_global_id], df_ddepths_sketch_shared[tid]);
                        }
		}
	}
}

void BACKWARD::preprocessSketchJacobian(
        const int sketch_mode,
        const int sketch_dim,
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float* projmatrix_raw,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
        const int selected_gaussian_size,
        const int* selected_gaussian_indices,
        const bool* selected_gaussian_bools,
	const float3* df_dmean2D,
	const float* df_dconic,
	glm::vec3* df_dmean3D,
	float* df_dcolor,
	float* df_ddepth,
	float* df_dcov3D,
	float* df_dsh,
	glm::vec3* df_dscale,
	glm::vec4* df_drot,
	float* df_dtau,
        const float* df_dopacity)
{
        int threads_per_block = 256;
        int x_dim = sketch_dim <= 1? 1 :
                   sketch_dim <= 2? 2 :
                   sketch_dim <= 4? 4 :
                   sketch_dim <= 8? 8 :
                   sketch_dim <= 16? 16 : 32;

        int y_dim = threads_per_block / x_dim;

        // We have a grid of sketch_dim x P threads. We group it so that each block
        // has roughly 256 threads
        dim3 block_dim(x_dim, y_dim, 1);
        dim3 grid_dim((sketch_dim + x_dim - 1) / x_dim, (P + y_dim - 1) / y_dim, 1);

        // Propagate gradients for the path of 2D conic matrix computation. 
        // Somewhat long, thus it is its own kernel rather than being part of 
        // "preprocess". When done, loss gradient w.r.t. 3D means has been
        // modified and gradient w.r.t. 3D covariance matrix has been computed.	
        computeCov2DSketchJacobianCUDA <<< grid_dim, block_dim >>> (
            sketch_mode,
            sketch_dim,
            P,
            means3D,
            radii,
            cov3Ds,
            focal_x,
            focal_y,
            tan_fovx,
            tan_fovy,
            viewmatrix,
            selected_gaussian_size,
            selected_gaussian_indices,
            selected_gaussian_bools,
            df_dconic,
            (float3*)df_dmean3D,
            df_dcov3D,
            df_dtau);

        // Propagate gradients for remaining steps: finish 3D mean gradients,
        // propagate color gradients to SH (if desireD), propagate 3D covariance
        // matrix gradients to scale and rotation.
        preprocessSketchJacobianCUDA<NUM_CHANNELS> <<< grid_dim, block_dim >>> (
            sketch_mode,
            sketch_dim,
            P, D, M,
            (float3*)means3D,
            radii,
            shs,
            clamped,
            (glm::vec3*)scales,
            (glm::vec4*)rotations,
            scale_modifier,
            viewmatrix,
            projmatrix,
            projmatrix_raw,
            campos,
            selected_gaussian_size,
            selected_gaussian_indices,
            selected_gaussian_bools,
            (float3*)df_dmean2D,
            (glm::vec3*)df_dmean3D,
            df_dcolor,
            df_ddepth,
            df_dcov3D,
            df_dsh,
            df_dscale,
            df_drot,
            df_dtau);
}

void BACKWARD::renderSketchJacobian(
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
	float* df_ddepths)
{

    renderSketchJacobianCUDA<NUM_CHANNELS> << <grid, block >> >(
            sketch_mode,
            sketch_dim,
            sketch_indices,
            ranges,
            point_list,
            W, H,
            P,
            bg_color,
            means2D,
            conic_opacity,
            colors,
            depths,
            final_Ts,
            n_contrib,
            selected_gaussian_size,
            selected_gaussian_indices,
            selected_gaussian_bools,
            df_dpixels,
            df_dpixels_depth,
            df_dmean2D,
            df_dconic2D,
            df_dopacity,
            df_dcolors,
            df_ddepths);

}
