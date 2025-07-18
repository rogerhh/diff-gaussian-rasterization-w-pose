#include "auxiliary.h"

#include <gtest/gtest.h>
#include <iostream>
#include <tuple>
#include <utility>
#include <torch/torch.h>

#include "test_utils.h"
#include "float_grad.h"
#include "helper_math.h"
#include "rasterizer.h"

#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#define BLOCK_X 16
#define BLOCK_Y 16

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

__global__
void compute_jvp(int P, 
                 float eps,
                 float* arg_grad,
                 bool* ref_grad_valid,
                 float* out_color, 
                 float* out_color_plus,
                 float* out_color_minus,
                 float* out_color_grad,
                 float* out_depth,
                 float* out_depth_plus,
                 float* out_depth_minus,
                 float* out_depth_grad,
                 float* out_opacity,
                 float* out_opacity_plus,
                 float* out_opacity_minus,
                 float* out_opacity_grad) {
    const double diff_eps = 1e-5;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    if (idx >= 13 && idx <= 20) {
        printf("Computing JVP for pixel %d\n", idx);
        printf("out_depth_plus[%d] = %f, out_depth[%d] = %f, out_depth_minus[%d] = %f\n",
               idx, out_depth_plus[idx], idx, out_depth[idx], idx, out_depth_minus[idx]);
        printf("out_opacity_plus[%d] = %f, out_opacity[%d] = %f, out_opacity_minus[%d] = %f\n",
               idx, out_opacity_plus[idx], idx, out_opacity[idx], idx, out_opacity_minus[idx]);
    }

    double shared_scale = 1.0 / (2.0 * (double) eps) * ((double) *arg_grad);

    for (int i = 0; i < 3; i++) {
        double diff1 = ((double) out_color_plus[idx * 3 + i]) - ((double) out_color[idx * 3 + i]);
        double diff2 = ((double) out_color[idx * 3 + i]) - ((double) out_color_minus[idx * 3 + i]);
        if (fabs(diff1 - diff2) > diff_eps) {
            ref_grad_valid[idx * 3 + i] = false;
        }

        double diff = ((double) out_color_plus[idx * 3 + i]) - ((double) out_color_minus[idx * 3 + i]);
        out_color_grad[idx * 3 + i] += diff * shared_scale;
    }

    double diff1 = ((double) out_depth_plus[idx]) - ((double) out_depth[idx]);
    double diff2 = ((double) out_depth[idx]) - ((double) out_depth_minus[idx]);
    if (fabs(diff1 - diff2) > diff_eps) {
        ref_grad_valid[idx] = false;
    }
    double diff_depth = ((double) out_depth_plus[idx]) - ((double) out_depth_minus[idx]);
    out_depth_grad[idx] += diff_depth * shared_scale;

    diff1 = ((double) out_opacity_plus[idx]) - ((double) out_opacity[idx]);
    diff2 = ((double) out_opacity[idx]) - ((double) out_opacity_minus[idx]);
    if (fabs(diff1 - diff2) > diff_eps) {
        ref_grad_valid[idx] = false;
    }

    double diff_opacity = ((double) out_opacity_plus[idx]) - ((double) out_opacity_minus[idx]);
    out_opacity_grad[idx] += diff_opacity * shared_scale;
}

void reset_args(int P, 
                int* n_touched) {
    cudaMemset(n_touched, 0, P * sizeof(int));
}

void call_forward_jvp(
        std::function<char* (size_t)> geometryBuffer,
        std::function<char* (size_t)> binningBuffer,
        std::function<char* (size_t)> imageBuffer,
        int P, int D, int M,
        float* background,
        int width, int height,
        float* means3D,
        float* shs,
        float* colors_precomp,
        float* opacities,
        float* scales,
        float scale_modifier,
        float* rotations,
        float* cov3D_precomp,
        float* viewmatrix,
        float* projmatrix,
        float* cam_pos,
        float tan_fovx, float tan_fovy,
        bool prefiltered,
        float* viewmatrix_grad,
        float* projmatrix_grad,
        float* cam_pos_grad,
        // Everything below is output
        float* out_color,
        float* out_depth,
        float* out_opacity,
        int* radii,
        int* n_touched,
        float* out_color_grad,
        float* out_depth_grad,
        float* out_opacity_grad,
        float* out_color_plus,
        float* out_color_minus,
        float* out_depth_plus,
        float* out_depth_minus,
        float* out_opacity_plus,
        float* out_opacity_minus,
        int* radii_plus,
        int* radii_minus,
        bool* ref_grad_valid) {

    float eps = 1e-6f;

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

    bool debug = false;

    // Run this once first to get medium value, this is needed for the JVP
    // checking if gradient is valid
    reset_args(P, n_touched);
    CudaRasterizer::Rasterizer::forward(
        geometryBuffer,
        binningBuffer,
        imageBuffer,
        P, D, M,
        background,
        width, height,
        means3D,
        shs,
        colors_precomp,
        opacities,
        scales,
        scale_modifier,
        rotations,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        cam_pos,
        tan_fovx, tan_fovy,
        prefiltered,
        out_color,
        out_depth,
        out_opacity,
        radii,
        n_touched,
        debug);

    for (int i = 0; i < 35; i++) {
        float* arg = aligned_args[i];
        float* arg_grad = aligned_grad[i];
        float arg_old;

        update_arg<<<1, 1>>>(arg, eps, arg_old_device, 0);
        cudaDeviceSynchronize();

        reset_args(P, n_touched);

        CudaRasterizer::Rasterizer::forward(
            geometryBuffer,
            binningBuffer,
            imageBuffer,
            P, D, M,
            background,
            width, height,
            means3D,
            shs,
            colors_precomp,
            opacities,
            scales,
            scale_modifier,
            rotations,
            cov3D_precomp,
            viewmatrix,
            projmatrix,
            cam_pos,
            tan_fovx, tan_fovy,
            prefiltered,
            out_color_plus,
            out_depth_plus,
            out_opacity_plus,
            radii_plus,
            n_touched,
            debug);

        cudaDeviceSynchronize();

        update_arg<<<1, 1>>>(arg, eps, arg_old_device, 1);

        cudaDeviceSynchronize();

        reset_args(P, n_touched);

        CudaRasterizer::Rasterizer::forward(
            geometryBuffer,
            binningBuffer,
            imageBuffer,
            P, D, M,
            background,
            width, height,
            means3D,
            shs,
            colors_precomp,
            opacities,
            scales,
            scale_modifier,
            rotations,
            cov3D_precomp,
            viewmatrix,
            projmatrix,
            cam_pos,
            tan_fovx, tan_fovy,
            prefiltered,
            out_color_minus,
            out_depth_minus,
            out_opacity_minus,
            radii_minus,
            n_touched,
            debug);

        cudaDeviceSynchronize();

        update_arg<<<1, 1>>>(arg, eps, arg_old_device, 2);

        cudaDeviceSynchronize();

        int threads_per_block = 256;
        int blocks = (width * height + threads_per_block - 1) / threads_per_block;
        compute_jvp<<<blocks, threads_per_block>>>(
            P, eps, arg_grad, ref_grad_valid,
            out_color, out_color_plus, out_color_minus, out_color_grad,
            out_depth, out_depth_plus, out_depth_minus, out_depth_grad,
            out_opacity, out_opacity_plus, out_opacity_minus, out_opacity_grad
        );
        cudaDeviceSynchronize();

    }

    free_device(arg_old_device);

    reset_args(P, n_touched);

    CudaRasterizer::Rasterizer::forward(
        geometryBuffer,
        binningBuffer,
        imageBuffer,
        P, D, M,
        background,
        width, height,
        means3D,
        shs,
        colors_precomp,
        opacities,
        scales,
        scale_modifier,
        rotations,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        cam_pos,
        tan_fovx, tan_fovy,
        prefiltered,
        out_color,
        out_depth,
        out_opacity,
        radii,
        n_touched,
        debug);
}

void call_forward_floatgrad(
        std::function<char* (size_t)> geometryBuffer,
        std::function<char* (size_t)> binningBuffer,
        std::function<char* (size_t)> imageBuffer,
        int P, int D, int M,
        float* background,
        int width, int height,
        float* means3D,
        float* shs,
        FloatGradArray<float> colors_precomp,
        float* opacities,
        float* scales,
        float scale_modifier,
        float* rotations,
        FloatGradArray<float> cov3D_precomp,
        FloatGradArray<float> viewmatrix,
        FloatGradArray<float> projmatrix,
        FloatGradArray<float> cam_pos,
        float tan_fovx, float tan_fovy,
        bool prefiltered,
        // Everything below is output
        FloatGradArray<float> out_color,
        FloatGradArray<float> out_depth,
        FloatGradArray<float> out_opacity,
        int* radii,
        int* n_touched) {

    bool debug = false;

    CudaRasterizer::Rasterizer::forwardJvp(
        geometryBuffer,
        binningBuffer,
        imageBuffer,
        P, D, M,
        background,
        width, height,
        means3D,
        shs,
        colors_precomp,
        opacities,
        scales,
        scale_modifier,
        rotations,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        cam_pos,      
        tan_fovx, tan_fovy,
        prefiltered,
        out_color,
        out_depth,
        out_opacity,
        radii,
        n_touched,
        debug
    );

}


TEST(ForwardTest, ForwardJvp_test) {

    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    torch::Tensor geomBuffer = torch::empty({0}, options);
    torch::Tensor binningBuffer = torch::empty({0}, options);
    torch::Tensor imgBuffer = torch::empty({0}, options);
    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

    std::vector<float> background_host;
    int background_rows, background_cols;
    read_csv("data/bg.csv", background_host, background_rows, background_cols);

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

    std::vector<float> out_color_ref_host(W * H * 3, 0.0f);
    std::vector<float> out_depth_ref_host(W * H, 0.0f);
    std::vector<float> out_opacity_ref_host(W * H, 0.0f);
    std::vector<int> radii_ref_host(orig_points_rows, 0);
    std::vector<int> tiles_touched_ref_host(orig_points_rows, 0);
    bool prefiltered = read_scalar<bool>("data/prefiltered.csv");

    std::vector<float> out_color_grad_ref_host(W * H * 3, 0.0f);
    std::vector<float> out_depth_grad_ref_host(W * H, 0.0f);
    std::vector<float> out_opacity_grad_ref_host(W * H, 0.0f);

    std::vector<char> ref_grad_valid_host(W * H, 1);

    int P = orig_points_rows;
    int D = sh_degree;
    int M = shs_cols;

    float* background_device = host_to_device(background_host.data(), background_rows * background_cols);
    float* orig_points_device = host_to_device(orig_points_host.data(), orig_points_rows * orig_points_cols);
    glm::vec3* scales_device = host_to_device((glm::vec3*) scales_host.data(), scales_rows);
    glm::vec4* rotations_device = host_to_device((glm::vec4*) rotations_host.data(), rotations_rows);
    float* opacities_device = host_to_device(opacities_host.data(), opacities_rows);
    float* shs_device = host_to_device(shs_host.data(), shs_rows * shs_cols);
    float* cov3D_precomp_device = host_to_device(cov3D_precomp_host.data(), cov3D_precomp_rows * cov3D_precomp_cols);
    float* colors_precomp_device = host_to_device(colors_precomp_host.data(), colors_precomp_rows * colors_precomp_cols);
    float* viewmatrix_device = host_to_device(viewmatrix_host.data(), viewmatrix_rows * viewmatrix_cols);
    float* projmatrix_device = host_to_device(projmatrix_host.data(), projmatrix_rows * projmatrix_cols);
    glm::vec3* cam_pos_device = host_to_device((glm::vec3*) cam_pos_host.data(), cam_pos_rows);
    float* viewmatrix_grad_device = host_to_device(viewmatrix_grad_host.data(), viewmatrix_rows * viewmatrix_cols);
    float* projmatrix_grad_device = host_to_device(projmatrix_grad_host.data(), projmatrix_rows * projmatrix_cols);
    glm::vec3* cam_pos_grad_device = host_to_device((glm::vec3*) cam_pos_grad_host.data(), cam_pos_rows);

    float* out_color_ref_device = host_to_device(out_color_ref_host.data(), W * H * 3);
    float* out_depth_ref_device = host_to_device(out_depth_ref_host.data(), W * H);
    float* out_opacity_ref_device = host_to_device(out_opacity_ref_host.data(), W * H);
    int* radii_ref_device = host_to_device(radii_ref_host.data(), orig_points_rows);
    int* tiles_touched_ref_device = host_to_device(tiles_touched_ref_host.data(), orig_points_rows);
    float* out_color_grad_ref_device = host_to_device(out_color_grad_ref_host.data(), W * H * 3);
    float* out_depth_grad_ref_device = host_to_device(out_depth_grad_ref_host.data(), W * H);
    float* out_opacity_grad_ref_device = host_to_device(out_opacity_grad_ref_host.data(), W * H);

    // Some temporary arrays for the out_color, out_depth, out_opacity, radii
    float* out_color_plus_device = host_to_device(out_color_ref_host.data(), W * H * 3);
    float* out_color_minus_device = host_to_device(out_color_ref_host.data(), W * H * 3);
    float* out_depth_plus_device = host_to_device(out_depth_ref_host.data(), W * H);
    float* out_depth_minus_device = host_to_device(out_depth_ref_host.data(), W * H);
    float* out_opacity_plus_device = host_to_device(out_opacity_ref_host.data(), W * H);
    float* out_opacity_minus_device = host_to_device(out_opacity_ref_host.data(), W * H);
    int* radii_plus_device = host_to_device(radii_ref_host.data(), orig_points_rows);
    int* radii_minus_device = host_to_device(radii_ref_host.data(), orig_points_rows);
    bool* ref_grad_valid_device = host_to_device((bool*) ref_grad_valid_host.data(), W * H);

    cudaDeviceSynchronize();

    call_forward_jvp(
        geomFunc,
        binningFunc,
        imgFunc,
        P, D, M,
        background_device,
        W, H,
        orig_points_device,
        shs_device,
        colors_precomp_device,
        opacities_device,
        (float*) scales_device,
        scale_modifier,
        (float*) rotations_device,
        cov3D_precomp_device,
        viewmatrix_device,
        projmatrix_device,
        (float*) cam_pos_device,
        tan_fovx, tan_fovy,
        prefiltered,
        viewmatrix_grad_device,
        projmatrix_grad_device,
        (float*) cam_pos_grad_device,
        // Everything below is output
        out_color_ref_device,
        out_depth_ref_device,
        out_opacity_ref_device,
        radii_ref_device,
        tiles_touched_ref_device,
        out_color_grad_ref_device,
        out_depth_grad_ref_device,
        out_opacity_grad_ref_device,
        out_color_plus_device,
        out_color_minus_device,
        out_depth_plus_device,
        out_depth_minus_device,
        out_opacity_plus_device,
        out_opacity_minus_device,
        radii_plus_device,
        radii_minus_device,
        ref_grad_valid_device);

    cudaDeviceSynchronize();

    device_to_host(out_color_ref_host.data(), out_color_ref_device, W * H * 3);
    device_to_host(out_depth_ref_host.data(), out_depth_ref_device, W * H);
    device_to_host(out_opacity_ref_host.data(), out_opacity_ref_device, W * H);
    device_to_host(radii_ref_host.data(), radii_ref_device, orig_points_rows);
    device_to_host(tiles_touched_ref_host.data(), tiles_touched_ref_device, orig_points_rows);
    device_to_host(out_color_grad_ref_host.data(), out_color_grad_ref_device, W * H * 3);
    device_to_host(out_depth_grad_ref_host.data(), out_depth_grad_ref_device, W * H);
    device_to_host(out_opacity_grad_ref_host.data(), out_opacity_grad_ref_device, W * H);
    device_to_host((bool*) ref_grad_valid_host.data(), ref_grad_valid_device, W * H);

    cudaDeviceSynchronize();

    // for (int i = 0; i < 10; i++) {
    //     std::cout << "out_color_ref_host[" << i << "] = ";
    //     for (int j = 0; j < 3; j++) {
    //         std::cout << out_color_ref_host[i * 3 + j] << " ";
    //     }
    //     std::cout << "\tout_depth_ref_host[" << i << "] = " << out_depth_ref_host[i] << "\t";
    //     std::cout << "out_opacity_ref_host[" << i << "] = " << out_opacity_ref_host[i] << "\n";
    //     std::cout << "out_color_grad_ref_host[" << i << "] = ";
    //     for (int j = 0; j < 3; j++) {
    //         std::cout << out_color_grad_ref_host[i * 3 + j] << " ";
    //     }
    //     std::cout << "\tout_depth_grad_ref_host[" << i << "] = " << out_depth_grad_ref_host[i] << "\t";
    //     std::cout << "out_opacity_grad_ref_host[" << i << "] = " << out_opacity_grad_ref_host[i] << "\n";

    // }

    std::vector<float> out_color_host(W * H * 3, 0.0f);
    std::vector<float> out_depth_host(W * H, 0.0f);
    std::vector<float> out_opacity_host(W * H, 0.0f);
    std::vector<int> radii_host(orig_points_rows, 0);
    std::vector<int> tiles_touched_host(orig_points_rows, 0);
    std::vector<float> out_color_grad_host(W * H * 3, 0.0f);
    std::vector<float> out_depth_grad_host(W * H, 0.0f);
    std::vector<float> out_opacity_grad_host(W * H, 0.0f);

    float* out_color_device = host_to_device(out_color_host.data(), W * H * 3);
    float* out_depth_device = host_to_device(out_depth_host.data(), W * H);
    float* out_opacity_device = host_to_device(out_opacity_host.data(), W * H);
    int* radii_device = host_to_device(radii_host.data(), orig_points_rows);
    int* tiles_touched_device = host_to_device(tiles_touched_host.data(), orig_points_rows);
    float* out_color_grad_device = host_to_device(out_color_grad_host.data(), W * H * 3);
    float* out_depth_grad_device = host_to_device(out_depth_grad_host.data(), W * H);
    float* out_opacity_grad_device = host_to_device(out_opacity_grad_host.data(), W * H);

    assert(colors_precomp_device == nullptr);
    assert(cov3D_precomp_device == nullptr);

    FloatGradArray<float> colors_precomp_floatgrad(colors_precomp_device, nullptr);
    FloatGradArray<float> cov3D_precomp_floatgrad(cov3D_precomp_device, nullptr);
    FloatGradArray<float> viewmatrix_floatgrad(viewmatrix_device, viewmatrix_grad_device);
    FloatGradArray<float> projmatrix_floatgrad(projmatrix_device, projmatrix_grad_device);
    FloatGradArray<float> cam_pos_floatgrad((float*) cam_pos_device, (float*) cam_pos_grad_device);
    FloatGradArray<float> out_color_floatgrad(out_color_device, out_color_grad_device);
    FloatGradArray<float> out_depth_floatgrad(out_depth_device, out_depth_grad_device);
    FloatGradArray<float> out_opacity_floatgrad(out_opacity_device, out_opacity_grad_device);

    call_forward_floatgrad(
        geomFunc,
        binningFunc,
        imgFunc,
        P, D, M,
        background_device,
        W, H,
        orig_points_device,
        shs_device,
        colors_precomp_floatgrad,
        opacities_device,
        (float*) scales_device,
        scale_modifier,
        (float*) rotations_device,
        cov3D_precomp_floatgrad,
        viewmatrix_floatgrad,
        projmatrix_floatgrad,
        cam_pos_floatgrad,
        tan_fovx, tan_fovy,
        prefiltered,
        // Everything below is output
        out_color_floatgrad,
        out_depth_floatgrad,
        out_opacity_floatgrad,
        radii_device,
        tiles_touched_device);

    cudaDeviceSynchronize();

    device_to_host(out_color_host.data(), out_color_device, W * H * 3);
    device_to_host(out_depth_host.data(), out_depth_device, W * H);
    device_to_host(out_opacity_host.data(), out_opacity_device, W * H);
    device_to_host(radii_host.data(), radii_device, orig_points_rows);
    device_to_host(tiles_touched_host.data(), tiles_touched_device, orig_points_rows);
    device_to_host(out_color_grad_host.data(), out_color_grad_device, W * H * 3);
    device_to_host(out_depth_grad_host.data(), out_depth_grad_device, W * H);
    device_to_host(out_opacity_grad_host.data(), out_opacity_grad_device, W * H);

    cudaDeviceSynchronize();

    // for (int i = 0; i < orig_points_rows; i++) {
    //     EXPECT_EQ(radii_ref_host[i], radii_host[i]);
    //     EXPECT_EQ(tiles_touched_ref_host[i], tiles_touched_host[i]);
    // }

    // for (int i = 0; i < W * H; i++) {
    //     EXPECT_TRUE(expect_near(out_color_ref_host[i], out_color_host[i], 1e-5f));
    //     EXPECT_TRUE(expect_near(out_depth_ref_host[i], out_depth_host[i], 1e-5f));
    //     EXPECT_TRUE(expect_near(out_opacity_ref_host[i], out_opacity_host[i], 1e-5f));


    // }

    // int total_pixels = W * H;
    // int correct_grad_count = 0;
    // for (int i = 0; i < W * H; i++) {
    //     std::cout << "Checking pixel " << i << "\n";
    //     bool correct = true;
    //     if (!ref_grad_valid_host[i]) {
    //         correct = false;
    //     }
    //     else if (!expect_near(out_color_grad_ref_host[i], out_color_grad_host[i], 0.05, true)) {
    //         correct = false;
    //     }
    //     else if (!expect_near(out_depth_grad_ref_host[i], out_depth_grad_host[i], 0.05, true)) {
    //         correct = false;
    //     }
    //     else if (!expect_near(out_opacity_grad_ref_host[i], out_opacity_grad_host[i], 0.05, true)) {
    //         correct = false;
    //     }

    //     if (correct) {
    //         correct_grad_count++;
    //     }
    // }
    // printf("Correct color gradients: %d / %d\n", correct_grad_count, total_pixels);

    free_device(background_device);
    free_device(orig_points_device);
    free_device(scales_device);
    free_device(rotations_device);
    free_device(opacities_device);
    free_device(shs_device);
    free_device(cov3D_precomp_device);
    free_device(colors_precomp_device);
    free_device(viewmatrix_device);
    free_device(projmatrix_device);
    free_device(cam_pos_device);
    free_device(viewmatrix_grad_device);
    free_device(projmatrix_grad_device);
    free_device(cam_pos_grad_device);
    free_device(out_color_ref_device);
    free_device(out_depth_ref_device);
    free_device(out_opacity_ref_device);
    free_device(radii_ref_device);
    free_device(tiles_touched_ref_device);
    free_device(out_color_grad_ref_device);
    free_device(out_depth_grad_ref_device);
    free_device(out_opacity_grad_ref_device);
    free_device(out_color_plus_device);
    free_device(out_color_minus_device);
    free_device(out_depth_plus_device);
    free_device(out_depth_minus_device);
    free_device(out_opacity_plus_device);
    free_device(out_opacity_minus_device);
    free_device(radii_plus_device);
    free_device(radii_minus_device);
    free_device(ref_grad_valid_device);
    free_device(out_color_device);
    free_device(out_depth_device);
    free_device(out_opacity_device);
    free_device(radii_device);
    free_device(tiles_touched_device);
    free_device(out_color_grad_device);
    free_device(out_depth_grad_device);
    free_device(out_opacity_grad_device);

}
