#include "auxiliary.h"

#include <gtest/gtest.h>
#include <iostream>
#include <tuple>
#include <utility>
#include <torch/torch.h>

#include "test_utils.h"
#include "float_grad.h"
#include "helper_math.h"
#include "forward.h"

#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#define BLOCK_X 16
#define BLOCK_Y 16

__global__
void compute_jvp(int W, int H, 
                 float eps,
                 float* arg_grad,
                 bool* ref_grad_valid,
                 float* final_T,
                 float* final_T_plus,
                 float* final_T_minus,
                 float* final_T_grad,
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

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;
    float shared_scale = 1.0 / (2.0 * (float) eps) * ((float) *arg_grad);

    if (idx == 1) {
        printf("JVP idx: %d, arg_grad: %f, final_T_plus = %f, final_T_minus = %f\n",
               idx, *arg_grad, final_T_plus[idx], final_T_minus[idx]);
    }

    float diff_final_T = ((float) final_T_plus[idx]) - ((float) final_T_minus[idx]);
    final_T_grad[idx] += diff_final_T * shared_scale;

    for (int i = 0; i < 3; i++) {
        float diff = ((float) out_color_plus[idx * 3 + i]) - ((float) out_color_minus[idx * 3 + i]);
        out_color_grad[idx * 3 + i] += diff * shared_scale;
    }

    float diff_depth = ((float) out_depth_plus[idx]) - ((float) out_depth_minus[idx]);
    out_depth_grad[idx] += diff_depth * shared_scale;

    float diff_opacity = ((float) out_opacity_plus[idx]) - ((float) out_opacity_minus[idx]);
    out_opacity_grad[idx] += diff_opacity * shared_scale;
}

void reset_args(int P, 
                int* n_touched) {
    cudaMemset(n_touched, 0, P * sizeof(int));
}

void call_render_jvp(
        int P,
        const dim3 grid, const dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        int W, int H,
        float2* means2D,
        float* colors,
        float4* conic_opacity,
        float* final_T,             // Output
        uint32_t* n_contrib,
        float* bg_color,
        float* out_color,           // Output 
        float* depth,
        float* out_depth,           // Output
        float* out_opacity,         // Output
        int* n_touched,             // Output
        std::vector<int> jvp_indices,
        float2* means2D_grad,
        float* colors_grad,
        float4* conic_opacity_grad,
        float* final_T_grad,        // Output
        float* out_color_grad,      // Output
        float* depth_grad,
        float* out_depth_grad,      // Output
        float* out_opacity_grad,    // Output
        // Temporary arrays for JVP
        float* final_T_plus,         
        float* final_T_minus,
        float* out_color_plus,
        float* out_color_minus,
        float* out_depth_plus,
        float* out_depth_minus,
        float* out_opacity_plus,
        float* out_opacity_minus,
        int* n_touched_plus,
        int* n_touched_minus,
        bool* ref_grad_valid) {

    float eps = 1e-1f;
    int jvp_args_len = 2 + 3 + 4 + 1; // means2D, colors, conic_opacity, depth

    int num_args = jvp_indices.size() * jvp_args_len;

    std::vector<float*> aligned_args(num_args);
    std::vector<float*> aligned_grad(num_args);

    std::vector<std::pair<float*, int>> args_in;
    std::vector<std::pair<float*, int>> grad_in;

    for (auto idx : jvp_indices) {
        args_in.push_back({(float*) &means2D[idx], 2});
        args_in.push_back({(float*) &colors[idx * 3], 3});
        args_in.push_back({(float*) &conic_opacity[idx], 4});
        args_in.push_back({(float*) &depth_grad[idx], 1});
        grad_in.push_back({(float*) &means2D_grad[idx], 2});
        grad_in.push_back({(float*) &colors_grad[idx * 3], 3});
        grad_in.push_back({(float*) &conic_opacity_grad[idx], 4});
        grad_in.push_back({(float*) &depth_grad[idx], 1});
    }

    align_params(args_in.data(), args_in.size(), aligned_args.data());
    align_params(grad_in.data(), grad_in.size(), aligned_grad.data());

    float arg_old_host;
    float* arg_old_device = host_to_device(&arg_old_host, 1);

    bool debug = false;

    // Run this once first to get medium value, this is needed for the JVP
    // checking if gradient is valid
    reset_args(P, n_touched);

    for (int i = 0; i < num_args; i++) {
        float* arg = aligned_args[i];
        float* arg_grad = aligned_grad[i];
        float arg_old;

        update_arg<<<1, 1>>>(arg, eps, arg_old_device, 0);
        cudaDeviceSynchronize();

        reset_args(P, n_touched);

        FORWARD::render(
            grid, block,
            ranges, point_list,
            W, H,
            means2D, colors, conic_opacity,
            final_T_plus, n_contrib,
            bg_color,
            out_color_plus, depth, out_depth_plus, out_opacity_plus,
            n_touched);

        cudaDeviceSynchronize();

        update_arg<<<1, 1>>>(arg, eps, arg_old_device, 1);

        cudaDeviceSynchronize();

        reset_args(P, n_touched);
        FORWARD::render(
            grid, block,
            ranges, point_list,
            W, H,
            means2D, colors, conic_opacity,
            final_T_minus, n_contrib,
            bg_color,
            out_color_minus, depth, out_depth_minus, out_opacity_minus,
            n_touched);

        cudaDeviceSynchronize();

        update_arg<<<1, 1>>>(arg, eps, arg_old_device, 2);

        cudaDeviceSynchronize();

        int threads_per_block = 256;
        int blocks = (W * H + threads_per_block - 1) / threads_per_block;
        compute_jvp<<<blocks, threads_per_block>>>(
            W, H, eps, arg_grad, ref_grad_valid,
            final_T, final_T_plus, final_T_minus, final_T_grad,
            out_color, out_color_plus, out_color_minus, out_color_grad,
            out_depth, out_depth_plus, out_depth_minus, out_depth_grad,
            out_opacity, out_opacity_plus, out_opacity_minus, out_opacity_grad
        );
        cudaDeviceSynchronize();

    }

    free_device(arg_old_device);

    reset_args(P, n_touched);

    FORWARD::render(
        grid, block,
        ranges, point_list,
        W, H,
        means2D, colors, conic_opacity,
        final_T, n_contrib,
        bg_color,
        out_color, depth, out_depth, out_opacity,
        n_touched);

}

void call_render_floatgrad(
        int P,
        const dim3 grid, const dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        int W, int H,
        FloatGradArray<float2> means2D,
        FloatGradArray<float> colors,
        FloatGradArray<float4> conic_opacity,
        FloatGradArray<float> final_T,             // Output
        uint32_t* n_contrib,
        float* bg_color,
        FloatGradArray<float> out_color,           // Output 
        FloatGradArray<float> depth,
        FloatGradArray<float> out_depth,           // Output
        FloatGradArray<float> out_opacity,         // Output
        int* n_touched             // Output
        ) {
    FORWARD::renderJvp(
        grid, block,
        ranges, point_list,
        W, H,
        means2D, colors, conic_opacity,
        final_T, n_contrib,
        bg_color,
        out_color, depth, out_depth, out_opacity,
        n_touched);
}


TEST(ForwardTest, RenderTests) {

    std::vector<int> jvp_indices = {0, 1, 35, 408, 5157, 6666, 12120};

    std::vector<int> grid_vec;
    int grid_rows, grid_cols;
    read_csv<int>("render_data/tile_grid.csv", grid_vec, grid_rows, grid_cols);
    dim3 grid(grid_vec[0], grid_vec[1], grid_vec[2]);

    std::vector<int> block_vec;
    int block_rows, block_cols;
    read_csv<int>("render_data/block.csv", block_vec, block_rows, block_cols);
    dim3 block(block_vec[0], block_vec[1], block_vec[2]);

    std::vector<int> ranges_host;
    int ranges_rows, ranges_cols;
    read_csv<int>("render_data/img_ranges.csv", ranges_host, ranges_rows, ranges_cols);

    std::vector<int> point_list_host;
    int point_list_rows, point_list_cols;
    read_csv<int>("render_data/point_list.csv", point_list_host, point_list_rows, point_list_cols);

    int W = read_scalar<int>("render_data/width.csv");
    int H = read_scalar<int>("render_data/height.csv");

    std::vector<float> means2D_host;
    int means2D_rows, means2D_cols;
    read_csv<float>("render_data/means2D_data.csv", means2D_host, means2D_rows, means2D_cols);
    std::vector<float> means2D_grad_host;
    read_csv<float>("render_data/means2D_grad.csv", means2D_grad_host, means2D_rows, means2D_cols);
    int P = means2D_rows;

    std::vector<float> colors_host;
    int colors_rows, colors_cols;
    read_csv<float>("render_data/rgb_data.csv", colors_host, colors_rows, colors_cols);
    std::vector<float> colors_grad_host;
    read_csv<float>("render_data/rgb_grad.csv", colors_grad_host, colors_rows, colors_cols);

    std::vector<float> conic_opacity_host;
    int conic_opacity_rows, conic_opacity_cols;
    read_csv<float>("render_data/conic_opacity_data.csv", conic_opacity_host, conic_opacity_rows, conic_opacity_cols);
    std::vector<float> conic_opacity_grad_host;
    read_csv<float>("render_data/conic_opacity_grad.csv", conic_opacity_grad_host, conic_opacity_rows, conic_opacity_cols);

    std::vector<float> final_T_ref_host(W * H, 0.0f);
    std::vector<float> final_T_grad_ref_host(W * H, 0.0f);
    std::vector<uint32_t> n_contrib_ref_host(W * H, 0);

    std::vector<float> bg_color_host;
    int bg_color_rows, bg_color_cols;
    read_csv<float>("render_data/background.csv", bg_color_host, bg_color_rows, bg_color_cols);

    std::vector<float> out_color_ref_host(W * H * 3, 0.0f);
    std::vector<float> out_color_grad_ref_host(W * H * 3, 0.0f);

    std::vector<float> depth_host;
    int depth_rows, depth_cols;
    read_csv<float>("render_data/depth_data.csv", depth_host, depth_rows, depth_cols);
    std::vector<float> depth_grad_host;
    read_csv<float>("render_data/depth_grad.csv", depth_grad_host, depth_rows, depth_cols);

    std::vector<float> out_depth_ref_host(W * H, 0.0f);
    std::vector<float> out_depth_grad_ref_host(W * H, 0.0f);
    std::vector<float> out_opacity_ref_host(W * H, 0.0f);
    std::vector<float> out_opacity_grad_ref_host(W * H, 0.0f);
    std::vector<int> n_touched_ref_host(P, 0);
    std::vector<char> ref_grad_valid_host(W * H, 1);

    // Since we can't run all gradients, only pick a small list of gaussians 
    // to run gradient for
    std::vector<float> means2D_grad_copy(means2D_grad_host.size(), 0.0f);
    std::vector<float> colors_grad_copy(colors_grad_host.size(), 0.0f);
    std::vector<float> conic_opacity_grad_copy(conic_opacity_grad_host.size(), 0.0f);
    std::vector<float> depth_grad_copy(depth_grad_host.size(), 0.0f);
    for (auto idx : jvp_indices) {
        for (int i = 0; i < 2; i++) {
            means2D_grad_copy[idx * 2 + i] = means2D_grad_host[idx * 2 + i];
        }
        for (int i = 0; i < 3; i++) {
            colors_grad_copy[idx * 3 + i] = colors_grad_host[idx * 3 + i];
        }
        for (int i = 0; i < 4; i++) {
            conic_opacity_grad_copy[idx * 4 + i] = conic_opacity_grad_host[idx * 4 + i];
        }
        depth_grad_copy[idx] = depth_grad_host[idx];
    }
    means2D_grad_host = means2D_grad_copy;
    colors_grad_host = colors_grad_copy;
    conic_opacity_grad_host = conic_opacity_grad_copy;
    depth_grad_host = depth_grad_copy;


    uint2* ranges_device = host_to_device((uint2*) ranges_host.data(), ranges_rows);
    uint32_t* point_list_device = host_to_device((uint32_t*) point_list_host.data(), point_list_rows);
    float2* means2D_device = host_to_device((float2*) means2D_host.data(), means2D_rows);
    float2* means2D_grad_device = host_to_device((float2*) means2D_grad_host.data(), means2D_rows);
    float* colors_device = host_to_device(colors_host.data(), colors_rows * 3);
    float* colors_grad_device = host_to_device(colors_grad_host.data(), colors_rows * 3);
    float4* conic_opacity_device = host_to_device((float4*) conic_opacity_host.data(), conic_opacity_rows);
    float4* conic_opacity_grad_device = host_to_device((float4*) conic_opacity_grad_host.data(), conic_opacity_rows);
    float* final_T_ref_device = host_to_device(final_T_ref_host.data(), W * H);
    float* final_T_grad_ref_device = host_to_device(final_T_grad_ref_host.data(), W * H);
    uint32_t* n_contrib_ref_device = host_to_device(n_contrib_ref_host.data(), W * H);
    float* bg_color_device = host_to_device(bg_color_host.data(), 3);
    float* out_color_ref_device = host_to_device(out_color_ref_host.data(), W * H * 3);
    float* out_color_grad_ref_device = host_to_device(out_color_grad_ref_host.data(), W * H * 3);
    float* depth_device = host_to_device(depth_host.data(), P);
    float* depth_grad_device = host_to_device(depth_grad_host.data(), P);
    float* out_depth_ref_device = host_to_device(out_depth_ref_host.data(), W * H);
    float* out_depth_grad_ref_device = host_to_device(out_depth_grad_ref_host.data(), W * H);
    float* out_opacity_ref_device = host_to_device(out_opacity_ref_host.data(), W * H);
    float* out_opacity_grad_ref_device = host_to_device(out_opacity_grad_ref_host.data(), W * H);
    int* n_touched_ref_device = host_to_device(n_touched_ref_host.data(), P);

    // Some temporary arrays for the out_color, out_depth, out_opacity, radii
    float* final_T_plus_device = host_to_device(final_T_ref_host.data(), W * H);
    float* final_T_minus_device = host_to_device(final_T_ref_host.data(), W * H);
    float* out_color_plus_device = host_to_device(out_color_ref_host.data(), W * H * 3);
    float* out_color_minus_device = host_to_device(out_color_ref_host.data(), W * H * 3);
    float* out_depth_plus_device = host_to_device(out_depth_ref_host.data(), W * H);
    float* out_depth_minus_device = host_to_device(out_depth_ref_host.data(), W * H);
    float* out_opacity_plus_device = host_to_device(out_opacity_ref_host.data(), W * H);
    float* out_opacity_minus_device = host_to_device(out_opacity_ref_host.data(), W * H);
    int* n_touched_plus_device = host_to_device(n_touched_ref_host.data(), P);
    int* n_touched_minus_device = host_to_device(n_touched_ref_host.data(), P);

    bool* ref_grad_valid_device = host_to_device((bool*) ref_grad_valid_host.data(), W * H);

    cudaDeviceSynchronize();

    call_render_jvp(
        P,
        grid, block,
        ranges_device,
        point_list_device,
        W, H,
        means2D_device,
        colors_device,
        conic_opacity_device,
        final_T_ref_device,             // Output
        n_contrib_ref_device,           // Output 
        bg_color_device,
        out_color_ref_device,           // Output
        depth_device,
        out_depth_ref_device,           // Output
        out_opacity_ref_device,         // Output
        n_touched_ref_device,           // Output
        jvp_indices,
        means2D_grad_device,
        colors_grad_device,
        conic_opacity_grad_device,
        final_T_grad_ref_device,        // Output
        out_color_grad_ref_device,  // Output   
        depth_grad_device,
        out_depth_grad_ref_device,  // Output   
        out_opacity_grad_ref_device,  // Output 
        final_T_plus_device,         
        final_T_minus_device,
        out_color_plus_device,
        out_color_minus_device,
        out_depth_plus_device,
        out_depth_minus_device,
        out_opacity_plus_device,
        out_opacity_minus_device,
        n_touched_plus_device,
        n_touched_minus_device,
        ref_grad_valid_device);

    cudaDeviceSynchronize();

    device_to_host(final_T_ref_host.data(), final_T_ref_device, W * H);
    device_to_host(final_T_grad_ref_host.data(), final_T_grad_ref_device, W * H);
    device_to_host(out_color_ref_host.data(), out_color_ref_device, W * H * 3);
    device_to_host(out_color_grad_ref_host.data(), out_color_grad_ref_device, W * H * 3);
    device_to_host(out_depth_ref_host.data(), out_depth_ref_device, W * H);
    device_to_host(out_depth_grad_ref_host.data(), out_depth_grad_ref_device, W * H);
    device_to_host(out_opacity_ref_host.data(), out_opacity_ref_device, W * H);
    device_to_host(out_opacity_grad_ref_host.data(), out_opacity_grad_ref_device, W * H);
    device_to_host(n_touched_ref_host.data(), n_touched_ref_device, P);

    for (int i = 0; i < 10; i++) {
        std::cout << "Final T[" << i << "]: " << final_T_ref_host[i] << std::endl;
        std::cout << "Final T Grad[" << i << "]: " << final_T_grad_ref_host[i] << std::endl;
        std::cout << "Out Color[" << i << "]: ";
        for (int j = 0; j < 3; j++) {
            std::cout << out_color_ref_host[i * 3 + j] << " ";
        }
        std::cout << std::endl;
        std::cout << "Out Color Grad[" << i << "]: ";
        for (int j = 0; j < 3; j++) {
            std::cout << out_color_grad_ref_host[i * 3 + j] << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }

    std::vector<float> final_T_host(W * H, 0.0f);
    std::vector<float> final_T_grad_host(W * H, 0.0f);
    std::vector<uint32_t> n_contrib_host(W * H, 0);
    std::vector<float> out_color_host(W * H * 3, 0.0f);
    std::vector<float> out_color_grad_host(W * H * 3, 0.0f);
    std::vector<float> out_depth_host(W * H, 0.0f);
    std::vector<float> out_depth_grad_host(W * H, 0.0f);
    std::vector<float> out_opacity_host(W * H, 0.0f);
    std::vector<float> out_opacity_grad_host(W * H, 0.0f);
    std::vector<int> n_touched_host(P, 0);

    float* final_T_device = host_to_device(final_T_host.data(), W * H);
    float* final_T_grad_device = host_to_device(final_T_grad_host.data(), W * H);
    uint32_t* n_contrib_device = host_to_device(n_contrib_host.data(), W * H);
    float* out_color_device = host_to_device(out_color_host.data(), W * H * 3);
    float* out_color_grad_device = host_to_device(out_color_grad_host.data(), W * H * 3);
    float* out_depth_device = host_to_device(out_depth_host.data(), W * H);
    float* out_depth_grad_device = host_to_device(out_depth_grad_host.data(), W * H);
    float* out_opacity_device = host_to_device(out_opacity_host.data(), W * H);
    float* out_opacity_grad_device = host_to_device(out_opacity_grad_host.data(), W * H);
    int* n_touched_device = host_to_device(n_touched_host.data(), P);

    FloatGradArray<float2> means2D_floatgrad(means2D_device, means2D_grad_device);
    FloatGradArray<float> colors_floatgrad(colors_device, colors_grad_device);
    FloatGradArray<float4> conic_opacity_floatgrad(conic_opacity_device, conic_opacity_grad_device);
    FloatGradArray<float> final_T_floatgrad(final_T_device, final_T_grad_device);
    FloatGradArray<float> out_color_floatgrad(out_color_device, out_color_grad_device);
    FloatGradArray<float> depth_floatgrad(depth_device, depth_grad_device);
    FloatGradArray<float> out_depth_floatgrad(out_depth_device, out_depth_grad_device);
    FloatGradArray<float> out_opacity_floatgrad(out_opacity_device, out_opacity_grad_device);

    call_render_floatgrad(
            P,
            grid, block,
            ranges_device,
            point_list_device,
            W, H,
            means2D_floatgrad,
            colors_floatgrad,
            conic_opacity_floatgrad,
            final_T_floatgrad,             // Output
            n_contrib_device,           // Output
            bg_color_device,
            out_color_floatgrad,           // Output
            depth_floatgrad,
            out_depth_floatgrad,           // Output
            out_opacity_floatgrad,         // Output
            n_touched_device              // Output
    );

    cudaDeviceSynchronize();

    device_to_host(final_T_host.data(), final_T_device, W * H);
    device_to_host(final_T_grad_host.data(), final_T_grad_device, W * H);
    device_to_host(n_contrib_host.data(), n_contrib_device, W * H);
    device_to_host(out_color_host.data(), out_color_device, W * H * 3);
    device_to_host(out_color_grad_host.data(), out_color_grad_device, W * H * 3);
    device_to_host(out_depth_host.data(), out_depth_device, W * H);
    device_to_host(out_depth_grad_host.data(), out_depth_grad_device, W * H);
    device_to_host(out_opacity_host.data(), out_opacity_device, W * H);
    device_to_host(out_opacity_grad_host.data(), out_opacity_grad_device, W * H);
    device_to_host(n_touched_host.data(), n_touched_device, P);

    for (int i = 0; i < 10; i++) {
        std::cout << "Final T[" << i << "]: " << final_T_host[i] << std::endl;
        std::cout << "Final T Grad[" << i << "]: " << final_T_grad_host[i] << std::endl;
        std::cout << "Out Color[" << i << "]: ";
        for (int j = 0; j < 3; j++) {
            std::cout << out_color_host[i * 3 + j] << " ";
        }
        std::cout << std::endl;
        std::cout << "Out Color Grad[" << i << "]: ";
        for (int j = 0; j < 3; j++) {
            std::cout << out_color_grad_host[i * 3 + j] << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }


    free_device(ranges_device);
    free_device(point_list_device);
    free_device(means2D_device);
    free_device(means2D_grad_device);
    free_device(colors_device);
    free_device(colors_grad_device);
    free_device(conic_opacity_device);
    free_device(conic_opacity_grad_device);
    free_device(final_T_ref_device);
    free_device(final_T_grad_ref_device);
    free_device(n_contrib_ref_device);
    free_device(bg_color_device);
    free_device(out_color_ref_device);
    free_device(out_color_grad_ref_device);
    free_device(depth_device);
    free_device(depth_grad_device);
    free_device(out_depth_ref_device);
    free_device(out_depth_grad_ref_device);
    free_device(out_opacity_ref_device);
    free_device(out_opacity_grad_ref_device);
    free_device(n_touched_ref_device);
    free_device(final_T_plus_device);
    free_device(final_T_minus_device);
    free_device(out_color_plus_device);
    free_device(out_color_minus_device);
    free_device(out_depth_plus_device);
    free_device(out_depth_minus_device);
    free_device(out_opacity_plus_device);
    free_device(out_opacity_minus_device);
    free_device(n_touched_plus_device);
    free_device(n_touched_minus_device);
    free_device(ref_grad_valid_device);
    free_device(final_T_device);
    free_device(final_T_grad_device);
    free_device(out_color_device);
    free_device(out_color_grad_device);
    free_device(out_depth_device);
    free_device(out_depth_grad_device);
    free_device(out_opacity_device);
    free_device(out_opacity_grad_device);
    free_device(n_touched_device);

}
