#include "auxiliary.h"

#include <gtest/gtest.h>
#include <iostream>
#include <tuple>
#include <utility>
#include "test_utils.h"
#include "float_grad.h"
#include "helper_math.h"
#include "forward.h"

#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#define BLOCK_X 16
#define BLOCK_Y 16

// __global__
// void update_arg(float* arg, float eps, float* arg_old, int mode) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx == 0) {
//         if (mode == 0) {
//             *arg_old = *arg;
//             *arg += eps;
//         }
//         else if (mode == 1) {
//             *arg = *arg_old - eps;
//         }
//         else if (mode == 2) {
//             *arg = *arg_old;
//         }
//     }
// }

__global__
void compute_jvp(int P, 
                 float eps,
                 float* arg_grad,
                 int* radii_plus,
                 int* radii_minus,
                 float2* means2D_plus,
                 float2* means2D_minus,
                 float2* means2D_grad,
                 float* depths_plus,
                 float* depths_minus,
                 float* depths_grad,
                 float* cov3Ds_plus,
                 float* cov3Ds_minus,
                 float* cov3Ds_grad,
                 float* rgb_plus,
                 float* rgb_minus,
                 float* rgb_grad,
                 float4* conic_opacity_plus,
                 float4* conic_opacity_minus,
                 float4* conic_opacity_grad,
                 bool* ref_grad_valid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    // Because we are using the central difference method,
    // Some discrete points may not be equal.
    if (radii_plus[idx] != radii_minus[idx]) {
        ref_grad_valid[idx] = false;
    }

    means2D_grad[idx] += (means2D_plus[idx] - means2D_minus[idx]) / (2.0f * eps) * (*arg_grad);
    depths_grad[idx] += (depths_plus[idx] - depths_minus[idx]) / (2.0f * eps) * (*arg_grad);
    for (int i = 0; i < 6; i++) {
        cov3Ds_grad[idx * 6 + i] += (cov3Ds_plus[idx * 6 + i] - cov3Ds_minus[idx * 6 + i]) / (2.0f * eps) * (*arg_grad);
    }
    for (int i = 0; i < 3; i++) {
        rgb_grad[idx * 3 + i] += (rgb_plus[idx * 3 + i] - rgb_minus[idx * 3 + i]) / (2.0f * eps) * (*arg_grad);
    }
    conic_opacity_grad[idx] += (conic_opacity_plus[idx] - conic_opacity_minus[idx]) / (2.0f * eps) * (*arg_grad);
}

void call_preprocess_jvp(int P, int D, int M,
                         float* means3D,
                         glm::vec3* scales,
                         float scale_modifier,
                         glm::vec4* rotations,
                         float* opacities,
                         float* shs,
                         bool* clamped,     // output
                         float* cov3D_precomp,
                         float* colors_precomp,
                         float* viewmatrix,
                         float* projmatrix,
                         glm::vec3* cam_pos,
                         int W, int H,
                         float focal_x, float focal_y,
                         float tan_fovx, float tan_fovy,
                         float* viewmatrix_grad,
                         float* projmatrix_grad,
                         glm::vec3* cam_pos_grad,
                         // Everything below is output
                         int* radii,
                         float2* means2D,
                         float* depths,
                         float* cov3Ds,
                         float* rgb,
                         float4* conic_opacity,
                         dim3 grid,
                         uint32_t* tiles_touched,
                         bool prefiltered,
                         float2* means2D_grad,
                         float* depths_grad,
                         float* cov3Ds_grad,
                         float* rgb_grad,
                         float4* conic_opacity_grad,
                         int* radii_plus,
                         int* radii_minus,
                         float2* means2D_plus,
                         float2* means2D_minus,
                         float* depths_plus,
                         float* depths_minus,
                         float* cov3Ds_plus,
                         float* cov3Ds_minus,
                         float* rgb_plus,
                         float* rgb_minus,
                         float4* conic_opacity_plus,
                         float4* conic_opacity_minus,
                         bool* ref_grad_valid) {

    float eps = 1e-2f;

    float* aligned_args[35];
    float* aligned_grad[35];

    std::pair<float*, int> args_in[3] = {
        {viewmatrix, 16},
        {projmatrix, 16},
        {(float*) cam_pos, 3}
    };
    std::pair<float*, int> grad_in[3] = {
        {viewmatrix_grad, 16},
        {projmatrix_grad, 16},
        {(float*) cam_pos_grad, 3}
    };
    align_params(args_in, 3, aligned_args);
    align_params(grad_in, 3, aligned_grad);

    float arg_old_host;
    float* arg_old_device = host_to_device(&arg_old_host, 1);

    for (int i = 0; i < 35; i++) {
        float* arg = aligned_args[i];
        float* arg_grad = aligned_grad[i];
        float arg_old;

        update_arg<<<1, 1>>>(arg, eps, arg_old_device, 0);
        cudaDeviceSynchronize();

        FORWARD::preprocess(P, D, M,
                            means3D,
                            scales,
                            scale_modifier,
                            rotations,
                            opacities,
                            shs,
                            clamped,
                            cov3D_precomp,
                            colors_precomp,
                            viewmatrix,
                            projmatrix,
                            cam_pos,
                            W, H,
                            focal_x, focal_y,
                            tan_fovx, tan_fovy,
                            radii_plus,
                            means2D_plus,
                            depths_plus,
                            cov3Ds_plus,
                            rgb_plus,
                            conic_opacity_plus,
                            grid,
                            tiles_touched,
                            prefiltered);

        cudaDeviceSynchronize();

        update_arg<<<1, 1>>>(arg, eps, arg_old_device, 1);

        cudaDeviceSynchronize();

        FORWARD::preprocess(P, D, M,
                            means3D,
                            scales,
                            scale_modifier,
                            rotations,
                            opacities,
                            shs,
                            clamped,
                            cov3D_precomp,
                            colors_precomp,
                            viewmatrix,
                            projmatrix,
                            cam_pos,
                            W, H,
                            focal_x, focal_y,
                            tan_fovx, tan_fovy,
                            radii_minus,
                            means2D_minus,
                            depths_minus,
                            cov3Ds_minus,
                            rgb_minus,
                            conic_opacity_minus,
                            grid,
                            tiles_touched,
                            prefiltered);

        cudaDeviceSynchronize();

        update_arg<<<1, 1>>>(arg, eps, arg_old_device, 2);

        cudaDeviceSynchronize();

        int threads_per_block = 256;
        int blocks = (P + threads_per_block - 1) / threads_per_block;
        compute_jvp<<<blocks, threads_per_block>>>(
            P, eps, arg_grad, radii_plus, radii_minus,
            means2D_plus, means2D_minus, means2D_grad,
            depths_plus, depths_minus, depths_grad,
            cov3Ds_plus, cov3Ds_minus, cov3Ds_grad,
            rgb_plus, rgb_minus, rgb_grad,
            conic_opacity_plus, conic_opacity_minus, conic_opacity_grad,
            ref_grad_valid
        );
        cudaDeviceSynchronize();

    }

    free_device(arg_old_device);

    FORWARD::preprocess(P, D, M,
                        means3D,
                        scales,
                        scale_modifier,
                        rotations,
                        opacities,
                        shs,
                        clamped,
                        cov3D_precomp,
                        colors_precomp,
                        viewmatrix,
                        projmatrix,
                        cam_pos,
                        W, H,
                        focal_x, focal_y,
                        tan_fovx, tan_fovy,
                        radii,
                        means2D,
                        depths,
                        cov3Ds,
                        rgb,
                        conic_opacity,
                        grid,
                        tiles_touched,
                        prefiltered);
}

void call_preprocess_floatgrad(int P, int D, int M,
                         float* means3D,
                         glm::vec3* scales,
                         float scale_modifier,
                         glm::vec4* rotations,
                         float* opacities,
                         float* shs,
                         bool* clamped,     // output
                         FloatGradArray<float> cov3D_precomp,
                         FloatGradArray<float> colors_precomp,
                         FloatGradArray<float> viewmatrix,
                         FloatGradArray<float> projmatrix,
                         FloatGradArray<glm::vec3> cam_pos,
                         int W, int H,
                         float focal_x, float focal_y,
                         float tan_fovx, float tan_fovy,
                         // Everything below is output
                         int* radii,
                         FloatGradArray<float2> means2D,
                         FloatGradArray<float> depths,
                         FloatGradArray<float> cov3Ds,
                         FloatGradArray<float> rgb,
                         FloatGradArray<float4> conic_opacity,
                         dim3 grid,
                         uint32_t* tiles_touched,
                         bool prefiltered) {
    FORWARD::preprocessJvp(P, D, M,
                        means3D,
                        scales,
                        scale_modifier,
                        rotations,
                        opacities,
                        shs,
                        clamped,
                        cov3D_precomp,
                        colors_precomp,
                        viewmatrix,
                        projmatrix,
                        cam_pos,
                        W, H,
                        focal_x, focal_y,
                        tan_fovx, tan_fovy,
                        radii,
                        means2D,
                        depths,
                        cov3Ds,
                        rgb,
                        conic_opacity,
                        grid,
                        tiles_touched,
                        prefiltered);
}


TEST(ForwardTest, PreprocessJvp_test) {
    std::vector<float> orig_points_host;
    int orig_points_rows, orig_points_cols;
    read_csv("data/means3D.csv", orig_points_host, orig_points_rows, orig_points_cols);
    EXPECT_EQ(orig_points_cols, 3);

    std::vector<float> scales_host;
    int scales_rows, scales_cols;
    read_csv("data/scales.csv", scales_host, scales_rows, scales_cols);

    float scale_modifier = read_scalar<float>("data/scale_modifier.csv");

    std::vector<float> rotations_host;
    int rotations_rows, rotations_cols;
    read_csv("data/rotations.csv", rotations_host, rotations_rows, rotations_cols);

    std::vector<float> opacities_host;
    int opacities_rows, opacities_cols;
    read_csv("data/opacities.csv", opacities_host, opacities_rows, opacities_cols);

    std::vector<float> shs_host;
    int shs_rows, shs_cols;
    read_csv("data/sh.csv", shs_host, shs_rows, shs_cols);

    bool* clamped_ref_host = new bool[orig_points_rows * 3];

    int sh_degree = read_scalar<int>("data/sh_degree.csv");

    std::vector<float> cov3D_precomp_host;
    int cov3D_precomp_rows, cov3D_precomp_cols;
    read_csv("data/cov3Ds_precomp.csv", cov3D_precomp_host, cov3D_precomp_rows, cov3D_precomp_cols);

    std::vector<float> colors_precomp_host;
    int colors_precomp_rows, colors_precomp_cols;
    read_csv("data/colors_precomp.csv", colors_precomp_host, colors_precomp_rows, colors_precomp_cols);

    std::vector<float> viewmatrix_host;
    int viewmatrix_rows, viewmatrix_cols;
    read_csv("data/viewmatrix.csv", viewmatrix_host, viewmatrix_rows, viewmatrix_cols);

    std::vector<float> projmatrix_host;
    int projmatrix_rows, projmatrix_cols;
    read_csv("data/projmatrix.csv", projmatrix_host, projmatrix_rows, projmatrix_cols);

    std::vector<float> cam_pos_host;
    int cam_pos_rows, cam_pos_cols;
    read_csv("data/campos.csv", cam_pos_host, cam_pos_rows, cam_pos_cols);

    int W = read_scalar<int>("data/image_width.csv");
    int H = read_scalar<int>("data/image_height.csv");

    float tan_fovx = read_scalar<float>("data/tanfovx.csv");
    float tan_fovy = read_scalar<float>("data/tanfovy.csv");
    float focal_y = H / (2.0f * tan_fovy);
    float focal_x = W / (2.0f * tan_fovx);

    std::vector<float> viewmatrix_grad_host(viewmatrix_rows * viewmatrix_cols, 0.0f);
    for (int i = 0; i < viewmatrix_rows * viewmatrix_cols; i++) {
        viewmatrix_grad_host[i] = (i % 4) * 0.03 + (i + 2) * 0.01f;
    }
    std::vector<float> projmatrix_grad_host(projmatrix_rows * projmatrix_cols, 0.0f);
    for (int i = 0; i < projmatrix_rows * projmatrix_cols; i++) {
        projmatrix_grad_host[i] = (i % 4) * 0.02 + (i + 2) * 0.02f;
    }
    std::vector<float> cam_pos_grad_host;
    for (int i = 0; i < cam_pos_rows * cam_pos_cols; i++) {
        cam_pos_grad_host.push_back((i * 0.014f) + 0.03f);
    }

    std::vector<int> radii_ref_host(orig_points_rows, 0);
    std::vector<float2> means2D_ref_host(orig_points_rows, {0.0f, 0.0f});
    std::vector<float> depths_ref_host(orig_points_rows, 0.0f);
    std::vector<float> cov3Ds_ref_host(orig_points_rows * 6, 0.0f);
    std::vector<float> rgb_ref_host(orig_points_rows * 3, 0.0f);
    std::vector<float4> conic_opacity_ref_host(orig_points_rows, {0.0f, 0.0f, 0.0f, 0.0f});
    dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    std::vector<uint32_t> tiles_touched_ref_host(orig_points_rows, 0);
    bool prefiltered = read_scalar<bool>("data/prefiltered.csv");
    std::vector<float2> means2D_grad_ref_host(orig_points_rows, {0.0f, 0.0f});
    std::vector<float> depths_grad_ref_host(orig_points_rows, 0.0f);
    std::vector<float> cov3Ds_grad_ref_host(orig_points_rows * 6, 0.0f);
    std::vector<float> rgb_grad_ref_host(orig_points_rows * 3, 0.0f);
    std::vector<float4> conic_opacity_grad_ref_host(orig_points_rows, {0.0f, 0.0f, 0.0f, 0.0f});
    bool* ref_grad_valid_host = new bool[orig_points_rows];
    for (int i = 0; i < orig_points_rows; i++) {
        ref_grad_valid_host[i] = true;
    }

    int P = orig_points_rows;
    int D = sh_degree;
    int M = shs_cols;

    float* orig_points_device = host_to_device(orig_points_host.data(), orig_points_rows * orig_points_cols);
    glm::vec3* scales_device = host_to_device((glm::vec3*) scales_host.data(), scales_rows);
    glm::vec4* rotations_device = host_to_device((glm::vec4*) rotations_host.data(), rotations_rows);
    float* opacities_device = host_to_device(opacities_host.data(), opacities_rows);
    float* shs_device = host_to_device(shs_host.data(), shs_rows * shs_cols);
    bool* clamped_ref_device = host_to_device(clamped_ref_host, orig_points_rows * 3);
    float* cov3D_precomp_device = host_to_device(cov3D_precomp_host.data(), cov3D_precomp_rows * cov3D_precomp_cols);
    float* colors_precomp_device = host_to_device(colors_precomp_host.data(), colors_precomp_rows * colors_precomp_cols);
    float* viewmatrix_device = host_to_device(viewmatrix_host.data(), viewmatrix_rows * viewmatrix_cols);
    float* projmatrix_device = host_to_device(projmatrix_host.data(), projmatrix_rows * projmatrix_cols);
    glm::vec3* cam_pos_device = host_to_device((glm::vec3*) cam_pos_host.data(), cam_pos_rows);
    float* viewmatrix_grad_device = host_to_device(viewmatrix_grad_host.data(), viewmatrix_rows * viewmatrix_cols);
    float* projmatrix_grad_device = host_to_device(projmatrix_grad_host.data(), projmatrix_rows * projmatrix_cols);
    glm::vec3* cam_pos_grad_device = host_to_device((glm::vec3*) cam_pos_grad_host.data(), cam_pos_rows);
    int* radii_ref_device = host_to_device(radii_ref_host.data(), orig_points_rows);
    float2* means2D_ref_device = host_to_device(means2D_ref_host.data(), orig_points_rows);
    float* depths_ref_device = host_to_device(depths_ref_host.data(), orig_points_rows);
    float* cov3Ds_ref_device = host_to_device(cov3Ds_ref_host.data(), orig_points_rows * 6);
    float* rgb_ref_device = host_to_device(rgb_ref_host.data(), orig_points_rows * 3);
    float4* conic_opacity_ref_device = host_to_device(conic_opacity_ref_host.data(), orig_points_rows);
    uint32_t* tiles_touched_ref_device = host_to_device(tiles_touched_ref_host.data(), orig_points_rows);
    float2* means2D_grad_ref_device = host_to_device((float2*) means2D_ref_host.data(), orig_points_rows);
    float* depths_grad_ref_device = host_to_device(depths_ref_host.data(), orig_points_rows);
    float* cov3Ds_grad_ref_device = host_to_device(cov3Ds_ref_host.data(), orig_points_rows * 6);
    float* rgb_grad_ref_device = host_to_device(rgb_ref_host.data(), orig_points_rows * 3);
    float4* conic_opacity_grad_ref_device = host_to_device(conic_opacity_ref_host.data(), orig_points_rows);

    // Some temporary arrays for the means2D, depths, cov3Ds, rgb, conic_opacity
    int* radii_plus_device = host_to_device(radii_ref_host.data(), orig_points_rows);
    int* radii_minus_device = host_to_device(radii_ref_host.data(), orig_points_rows);
    float2* means2D_plus_device = host_to_device(means2D_ref_host.data(), orig_points_rows);
    float2* means2D_minus_device = host_to_device(means2D_ref_host.data(), orig_points_rows);
    float* depths_plus_device = host_to_device(depths_ref_host.data(), orig_points_rows);
    float* depths_minus_device = host_to_device(depths_ref_host.data(), orig_points_rows);
    float* cov3Ds_plus_device = host_to_device(cov3Ds_ref_host.data(), orig_points_rows * 6);
    float* cov3Ds_minus_device = host_to_device(cov3Ds_ref_host.data(), orig_points_rows * 6);
    float* rgb_plus_device = host_to_device(rgb_ref_host.data(), orig_points_rows * 3);
    float* rgb_minus_device = host_to_device(rgb_ref_host.data(), orig_points_rows * 3);
    float4* conic_opacity_plus_device = host_to_device(conic_opacity_ref_host.data(), orig_points_rows);
    float4* conic_opacity_minus_device = host_to_device(conic_opacity_ref_host.data(), orig_points_rows);
    bool* ref_grad_valid_device = host_to_device(ref_grad_valid_host, orig_points_rows);

    cudaDeviceSynchronize();

    call_preprocess_jvp(
        P, D, M,
        orig_points_device,
        scales_device,
        scale_modifier,
        rotations_device,
        opacities_device,
        shs_device,
        clamped_ref_device,
        cov3D_precomp_device,
        colors_precomp_device,
        viewmatrix_device,
        projmatrix_device,
        cam_pos_device,
        W, H,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        viewmatrix_grad_device,
        projmatrix_grad_device,
        cam_pos_grad_device,
        radii_ref_device,
        means2D_ref_device,
        depths_ref_device,
        cov3Ds_ref_device,
        rgb_ref_device,
        conic_opacity_ref_device,
        tile_grid,
        tiles_touched_ref_device,
        prefiltered,
        means2D_grad_ref_device,
        depths_grad_ref_device,
        cov3Ds_grad_ref_device,
        rgb_grad_ref_device,
        conic_opacity_grad_ref_device,
        radii_plus_device,
        radii_minus_device,
        means2D_plus_device,
        means2D_minus_device,
        depths_plus_device,
        depths_minus_device,
        cov3Ds_plus_device,
        cov3Ds_minus_device,
        rgb_plus_device,
        rgb_minus_device,
        conic_opacity_plus_device,
        conic_opacity_minus_device,
        ref_grad_valid_device
    );

    cudaDeviceSynchronize();

    device_to_host(clamped_ref_host, clamped_ref_device, orig_points_rows * 3);
    device_to_host(radii_ref_host.data(), radii_ref_device, orig_points_rows);
    device_to_host(means2D_ref_host.data(), means2D_ref_device, orig_points_rows);
    device_to_host(depths_ref_host.data(), depths_ref_device, orig_points_rows);
    device_to_host(cov3Ds_ref_host.data(), cov3Ds_ref_device, orig_points_rows * 6);
    device_to_host(rgb_ref_host.data(), rgb_ref_device, orig_points_rows * 3);
    device_to_host(conic_opacity_ref_host.data(), conic_opacity_ref_device, orig_points_rows);
    device_to_host(tiles_touched_ref_host.data(), tiles_touched_ref_device, orig_points_rows);
    device_to_host(means2D_grad_ref_host.data(), means2D_grad_ref_device, orig_points_rows);
    device_to_host(depths_grad_ref_host.data(), depths_grad_ref_device, orig_points_rows);
    device_to_host(cov3Ds_grad_ref_host.data(), cov3Ds_grad_ref_device, orig_points_rows * 6);
    device_to_host(rgb_grad_ref_host.data(), rgb_grad_ref_device, orig_points_rows * 3);
    device_to_host(conic_opacity_grad_ref_host.data(), conic_opacity_grad_ref_device, orig_points_rows);
    device_to_host(ref_grad_valid_host, ref_grad_valid_device, orig_points_rows);

    cudaDeviceSynchronize();

    bool* clamped_host = new bool[orig_points_rows * 3];
    std::vector<int> radii_host(orig_points_rows, 0);
    std::vector<float2> means2D_host(orig_points_rows, {0.0f, 0.0f});
    std::vector<float> depths_host(orig_points_rows, 0.0f);
    std::vector<float> cov3Ds_host(orig_points_rows * 6, 0.0f);
    std::vector<float> rgb_host(orig_points_rows * 3, 0.0f);
    std::vector<float4> conic_opacity_host(orig_points_rows, {0.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint32_t> tiles_touched_host(orig_points_rows, 0);
    std::vector<float> cov3D_precomp_grad_host(cov3D_precomp_rows * cov3D_precomp_cols, 0.0f);
    std::vector<float> color_precomp_grad_host(colors_precomp_rows * colors_precomp_cols, 0.0f);
    std::vector<float2> means2D_grad_host(orig_points_rows, {0.0f, 0.0f});
    std::vector<float> depths_grad_host(orig_points_rows, 0.0f);
    std::vector<float> cov3Ds_grad_host(orig_points_rows * 6, 0.0f);
    std::vector<float> rgb_grad_host(orig_points_rows * 3, 0.0f);
    std::vector<float4> conic_opacity_grad_host(orig_points_rows, {0.0f, 0.0f, 0.0f, 0.0f});

    bool* clamped_device = host_to_device(clamped_host, orig_points_rows * 3);
    int* radii_device = host_to_device(radii_host.data(), orig_points_rows);
    float2* means2D_device = host_to_device(means2D_host.data(), orig_points_rows);
    float* depths_device = host_to_device(depths_host.data(), orig_points_rows);
    float* cov3Ds_device = host_to_device(cov3Ds_host.data(), orig_points_rows * 6);
    float* rgb_device = host_to_device(rgb_host.data(), orig_points_rows * 3);
    float4* conic_opacity_device = host_to_device(conic_opacity_host.data(), orig_points_rows);
    uint32_t* tiles_touched_device = host_to_device(tiles_touched_host.data(), orig_points_rows);
    float* cov3D_precomp_grad_device = host_to_device(cov3D_precomp_grad_host.data(), cov3D_precomp_rows * cov3D_precomp_cols);
    float* colors_precomp_grad_device = host_to_device(color_precomp_grad_host.data(), colors_precomp_rows * colors_precomp_cols);
    float2* means2D_grad_device = host_to_device(means2D_grad_host.data(), orig_points_rows);
    float* depths_grad_device = host_to_device(depths_grad_host.data(), orig_points_rows);
    float* cov3Ds_grad_device = host_to_device(cov3Ds_grad_host.data(), orig_points_rows * 6);
    float* rgb_grad_device = host_to_device(rgb_grad_host.data(), orig_points_rows * 3);
    float4* conic_opacity_grad_device = host_to_device(conic_opacity_grad_host.data(), orig_points_rows);

    FloatGradArray<float> cov3D_precomp_floatgrad(cov3D_precomp_device, cov3D_precomp_grad_device);
    FloatGradArray<float> colors_precomp_floatgrad(colors_precomp_device, colors_precomp_grad_device);
    FloatGradArray<float> viewmatrix_floatgrad(viewmatrix_device, viewmatrix_grad_device);
    FloatGradArray<float> projmatrix_floatgrad(projmatrix_device, projmatrix_grad_device);
    FloatGradArray<glm::vec3> cam_pos_floatgrad(cam_pos_device, cam_pos_grad_device);
    FloatGradArray<float2> means2D_floatgrad(means2D_device, means2D_grad_device);
    FloatGradArray<float> depths_floatgrad(depths_device, depths_grad_device);
    FloatGradArray<float> cov3Ds_floatgrad(cov3Ds_device, cov3Ds_grad_device);
    FloatGradArray<float> rgb_floatgrad(rgb_device, rgb_grad_device);
    FloatGradArray<float4> conic_opacity_floatgrad(conic_opacity_device, conic_opacity_grad_device);

    cudaDeviceSynchronize();

    call_preprocess_floatgrad(P, D, M,
            orig_points_device,
            scales_device,
            scale_modifier,
            rotations_device,
            opacities_device,
            shs_device,
            clamped_device,
            cov3D_precomp_floatgrad,
            colors_precomp_floatgrad,
            viewmatrix_floatgrad,
            projmatrix_floatgrad,
            cam_pos_floatgrad,
            W, H,
            focal_x, focal_y,
            tan_fovx, tan_fovy,
            radii_device,
            means2D_floatgrad,
            depths_floatgrad,
            cov3Ds_floatgrad,
            rgb_floatgrad,
            conic_opacity_floatgrad,
            tile_grid,
            tiles_touched_device,
            prefiltered);

    cudaDeviceSynchronize();

    device_to_host(clamped_host, clamped_device, orig_points_rows * 3);
    device_to_host(radii_host.data(), radii_device, orig_points_rows);
    device_to_host(means2D_host.data(), means2D_device, orig_points_rows);
    device_to_host(depths_host.data(), depths_device, orig_points_rows);
    device_to_host(cov3Ds_host.data(), cov3Ds_device, orig_points_rows * 6);
    device_to_host(rgb_host.data(), rgb_device, orig_points_rows * 3);
    device_to_host(conic_opacity_host.data(), conic_opacity_device, orig_points_rows);
    device_to_host(tiles_touched_host.data(), tiles_touched_device, orig_points_rows);
    device_to_host(means2D_grad_host.data(), means2D_grad_device, orig_points_rows);
    device_to_host(depths_grad_host.data(), depths_grad_device, orig_points_rows);
    device_to_host(cov3Ds_grad_host.data(), cov3Ds_grad_device, orig_points_rows * 6);
    device_to_host(rgb_grad_host.data(), rgb_grad_device, orig_points_rows * 3);
    device_to_host(conic_opacity_grad_host.data(), conic_opacity_grad_device, orig_points_rows);

    cudaDeviceSynchronize();

    // for (int i = 45; i < 46; i++) {
    //     std::cout << "Splat: " << i << "\t"
    //               << "ref_grad valid: " << ref_grad_valid_host[i] << "\t"
    //               << "means2D ref: (" << means2D_ref_host[i].x << ", " << means2D_ref_host[i].y << ")\t"
    //               << "means2D ref grad: (" << means2D_grad_ref_host[i].x << ", " << means2D_grad_ref_host[i].y << ")\t"
    //               << "means2D: (" << means2D_host[i].x << ", " << means2D_host[i].y << ")\t"
    //               << "means2D grad: (" << means2D_grad_host[i].x << ", " << means2D_grad_host[i].y << ")" << std::endl;
    // }

    for (int i = 0; i < P; i++) {
        EXPECT_EQ(radii_host[i], radii_ref_host[i]);
        EXPECT_TRUE(expect_near(means2D_host[i], means2D_ref_host[i], 1e-1f));
        EXPECT_TRUE(expect_near(depths_host[i], depths_ref_host[i], 1e-1f));
        for (int j = 0; j < 6; j++) {
            EXPECT_TRUE(expect_near(cov3Ds_host[i * 6 + j], cov3Ds_ref_host[i * 6 + j], 1e-1f));
        }
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(expect_near(rgb_host[i * 3 + j], rgb_ref_host[i * 3 + j], 1e-1f)) << " at index i = " << i;
        }
        EXPECT_TRUE(expect_near(conic_opacity_host[i], conic_opacity_ref_host[i], 1e-1f));
        EXPECT_EQ(tiles_touched_host[i], tiles_touched_ref_host[i]);
        if (!ref_grad_valid_host[i]) {
            continue;
        }
        EXPECT_TRUE(expect_near(means2D_grad_host[i], means2D_grad_ref_host[i], 1e-1f)) 
            << " at index i = " << i;
        EXPECT_TRUE(expect_near(depths_grad_host[i], depths_grad_ref_host[i], 1e-1f)) << " at index i = " << i;
        for (int j = 0; j < 6; j++) {
            EXPECT_TRUE(expect_near(cov3Ds_grad_host[i * 6 + j], cov3Ds_grad_ref_host[i * 6 + j], 1e-1f)) << " at index i = " << i << ", j = " << j;
        }
        for (int j = 0; j < 3; j++) {
            EXPECT_TRUE(expect_near(rgb_grad_host[i * 3 + j], rgb_grad_ref_host[i * 3 + j], 1e-1f)) << " at index i = " << i << ", j = " << j;
        }
        EXPECT_TRUE(expect_near(conic_opacity_grad_host[i], conic_opacity_grad_ref_host[i], 1e-1f)) << " at index i = " << i;
    }
            

    free(clamped_ref_host);
    free(clamped_host);
    free(ref_grad_valid_host);
    free_device(orig_points_device);
    free_device(scales_device);
    free_device(rotations_device);
    free_device(opacities_device);
    free_device(shs_device);
    free_device(clamped_ref_device);
    free_device(cov3D_precomp_device);
    free_device(colors_precomp_device);
    free_device(viewmatrix_device);
    free_device(projmatrix_device);
    free_device(cam_pos_device);
    free_device(viewmatrix_grad_device);
    free_device(projmatrix_grad_device);
    free_device(cam_pos_grad_device);
    free_device(radii_ref_device);
    free_device(means2D_ref_device);
    free_device(depths_ref_device);
    free_device(cov3Ds_ref_device);
    free_device(rgb_ref_device);
    free_device(conic_opacity_ref_device);
    free_device(tiles_touched_ref_device);
    free_device(means2D_grad_ref_device);
    free_device(depths_grad_ref_device);
    free_device(cov3Ds_grad_ref_device);
    free_device(rgb_grad_ref_device);
    free_device(conic_opacity_grad_ref_device);

    free_device(clamped_device);
    free_device(radii_device);
    free_device(means2D_device);
    free_device(depths_device);
    free_device(cov3Ds_device);
    free_device(rgb_device);
    free_device(conic_opacity_device);
    free_device(tiles_touched_device);
    free_device(means2D_grad_device);
    free_device(depths_grad_device);
    free_device(cov3Ds_grad_device);
    free_device(rgb_grad_device);
    free_device(conic_opacity_grad_device);


}
