#include "auxiliary.h"

#include <gtest/gtest.h>
#include <iostream>
#include <tuple>
#include <utility>
#include "test_utils.h"
#include "float_grad.h"
#include "helper_math.h"

__global__
void call_in_frustum_jvp(int orig_points_rows,
                         float* orig_points, 
                         float* viewmatrix,
                         float* projmatrix,
                         bool prefiltered,
                         float3* p_view,
                         bool* in_frustum_ret,
                         float* viewmatrix_grad,
                         float* projmatrix_grad,
                         float3* p_view_grad) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= orig_points_rows) {
        return; // Out of bounds
    }

    in_frustum_ret[idx] = in_frustum(idx, orig_points, viewmatrix,
                                     projmatrix, prefiltered, p_view[idx]);

    p_view_grad[idx] = make_float3(0.0f, 0.0f, 0.0f);

    // Make local copies of the matrices
    float viewmatrix_local[16];
    float projmatrix_local[16];

    for (int i = 0; i < 16; i++) {
        viewmatrix_local[i] = viewmatrix[i];
        projmatrix_local[i] = projmatrix[i];
    }


    float eps = 1e-2f;

    float* aligned_args[32];
    float* aligned_grad[32];
    std::pair<float*, int> args_in[2] = {
        {viewmatrix_local, 16},
        {projmatrix_local, 16}
    };
    std::pair<float*, int> grad_in[2] = {
        {viewmatrix_grad, 16},
        {projmatrix_grad, 16}
    };
    align_params(args_in, 2, aligned_args);
    align_params(grad_in, 2, aligned_grad);

    for (int i = 0; i < 32; i++) {
        float* arg = aligned_args[i];
        float* arg_grad = aligned_grad[i];
        float arg_old = *arg;
        *arg += eps;
        float3 p_view_plus;
        in_frustum(idx, orig_points, viewmatrix_local,
                    projmatrix_local, prefiltered, p_view_plus);
        *arg = arg_old - eps;
        float3 p_view_minus;
        in_frustum(idx, orig_points, viewmatrix_local,
                    projmatrix_local, prefiltered, p_view_minus);
        p_view_grad[idx] += *arg_grad * (p_view_plus - p_view_minus) / (2.0f * eps);
        *arg = arg_old;
    }

}

__global__ 
void call_in_frustum_floatgrad(int orig_points_rows,
                               float* orig_points, 
                               FloatGradArray<const float> viewmatrix,
                               FloatGradArray<const float> projmatrix,
                               bool prefiltered,
                               FloatGradArray<float3> p_view,
                               bool* in_frustum_ret) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= orig_points_rows) {
        return; // Out of bounds
    }

    in_frustum_ret[idx] = in_frustum(idx, orig_points, viewmatrix,
                                   projmatrix, prefiltered, p_view[idx]);
}

TEST(ForwardTest, InFrustumTest) {
    std::vector<float> orig_points_host_scalar;
    int orig_points_rows, orig_points_cols;
    read_csv<float>("data/means3D.csv", orig_points_host_scalar, orig_points_rows, orig_points_cols);
    EXPECT_EQ(orig_points_cols, 3);
    float3* orig_points_host = reinterpret_cast<float3*>(orig_points_host_scalar.data());

    std::vector<float> viewmatrix_host;
    int viewmatrix_rows, viewmatrix_cols;
    read_csv<float>("data/viewmatrix.csv", viewmatrix_host, viewmatrix_rows, viewmatrix_cols);

    std::vector<float> projmatrix_host;
    int projmatrix_rows, projmatrix_cols;
    read_csv<float>("data/projmatrix.csv", projmatrix_host, projmatrix_rows, projmatrix_cols);

    std::vector<float> prefiltered_host;
    int prefiltered_rows, prefiltered_cols;
    read_csv<float>("data/prefiltered.csv", prefiltered_host, prefiltered_rows, prefiltered_cols);
    bool prefiltered = prefiltered_host[0] != 0.0f;

    // Set up gradients
    std::vector<float> viewmatrix_grad_host;
    std::vector<float> projmatrix_grad_host;
    for (int i = 0; i < viewmatrix_rows * viewmatrix_cols; i++) {
        viewmatrix_grad_host.push_back(i * 0.01f); 
        projmatrix_grad_host.push_back(i * 0.03f); 
    }

    float* orig_points_device = host_to_device((float*) orig_points_host, orig_points_rows * orig_points_cols);
    float* viewmatrix_device = host_to_device(viewmatrix_host.data(), viewmatrix_rows * viewmatrix_cols);
    float* projmatrix_device = host_to_device(projmatrix_host.data(), projmatrix_rows * projmatrix_cols);
    float* viewmatrix_grad_device = host_to_device(viewmatrix_grad_host.data(), viewmatrix_rows * viewmatrix_cols);
    float* projmatrix_grad_device = host_to_device(projmatrix_grad_host.data(), projmatrix_rows * projmatrix_cols);

    std::vector<float3> p_view_ref_host(orig_points_rows);
    std::vector<float3> p_view_ref_grad_host(orig_points_rows);
    std::vector<char> in_frustum_ret_ref_host(orig_points_rows, false);

    float3* p_view_ref_device = host_to_device(p_view_ref_host.data(), orig_points_rows);
    float3* p_view_ref_grad_device = host_to_device(p_view_ref_grad_host.data(), orig_points_rows);
    bool* in_frustum_ret_ref_device = host_to_device((bool*) in_frustum_ret_ref_host.data(), orig_points_rows);

    int num_blocks = (orig_points_rows + 255) / 256;

    call_in_frustum_jvp<<<num_blocks, 256>>>(
                                  orig_points_rows, 
                                  orig_points_device, 
                                  viewmatrix_device, 
                                  projmatrix_device, 
                                  prefiltered, 
                                  p_view_ref_device, 
                                  in_frustum_ret_ref_device, 
                                  viewmatrix_grad_device, 
                                  projmatrix_grad_device, 
                                  p_view_ref_grad_device);

    cudaDeviceSynchronize();

    cudaMemcpy(in_frustum_ret_ref_host.data(), in_frustum_ret_ref_device, orig_points_rows * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(p_view_ref_host.data(), p_view_ref_device, orig_points_rows * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(p_view_ref_grad_host.data(), p_view_ref_grad_device, orig_points_rows * sizeof(float3), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < 2; i++) {
    //     if (in_frustum_ret_ref_host[i]) {
    //         std::cout << "Point " << i << " is in frustum: "
    //                   << p_view_ref_host[i].x << ", "
    //                   << p_view_ref_host[i].y << ", "
    //                   << p_view_ref_host[i].z << ", "
    //                   << p_view_ref_grad_host[i].x << ", "
    //                   << p_view_ref_grad_host[i].y << ", "
    //                   << p_view_ref_grad_host[i].z << std::endl;

    //     } else {
    //         std::cout << "Point " << i << " is NOT in frustum." << std::endl;
    //     }
    // }

    std::vector<float3> p_view_host(orig_points_rows);
    std::vector<float3> p_view_grad_host(orig_points_rows);
    std::vector<char> in_frustum_ret_host(orig_points_rows, false);

    float3* p_view_device = host_to_device(p_view_host.data(), orig_points_rows);
    float3* p_view_grad_device = host_to_device(p_view_grad_host.data(), orig_points_rows);
    bool* in_frustum_ret_device = host_to_device((bool*) in_frustum_ret_host.data(), orig_points_rows);

    FloatGradArray<const float> viewmatrix_floatgrad(viewmatrix_device, viewmatrix_grad_device);
    FloatGradArray<const float> projmatrix_floatgrad(projmatrix_device, projmatrix_grad_device);
    FloatGradArray<float3> p_view_floatgrad(p_view_device, p_view_grad_device);


    call_in_frustum_floatgrad<<<num_blocks, 256>>>(
                               orig_points_rows, 
                               orig_points_device, 
                               viewmatrix_floatgrad, 
                               projmatrix_floatgrad, 
                               prefiltered, 
                               p_view_floatgrad, 
                               in_frustum_ret_device);

    cudaDeviceSynchronize();

    cudaMemcpy(in_frustum_ret_host.data(), in_frustum_ret_device, orig_points_rows * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(p_view_host.data(), p_view_device, orig_points_rows * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(p_view_grad_host.data(), p_view_grad_device, orig_points_rows * sizeof(float3), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < 2; i++) {
    //     if (in_frustum_ret_host[i]) {
    //         std::cout << "Point " << i << " is in frustum: "
    //                   << p_view_host[i].x << ", "
    //                   << p_view_host[i].y << ", "
    //                   << p_view_host[i].z << ", "
    //                   << p_view_grad_host[i].x << ", "
    //                   << p_view_grad_host[i].y << ", "
    //                   << p_view_grad_host[i].z << std::endl;
    //     } else {
    //         std::cout << "Point " << i << " is NOT in frustum." << std::endl;
    //     }
    // }

    for (int i = 0; i < orig_points_rows; i++) {
        if (in_frustum_ret_ref_host[i] != in_frustum_ret_host[i]) {
            std::cerr << "Mismatch at index " << i << ": "
                      << "ref: " << in_frustum_ret_ref_host[i] 
                      << ", test: " << in_frustum_ret_host[i] << std::endl;
        }
        EXPECT_EQ(in_frustum_ret_ref_host[i], in_frustum_ret_host[i]);
        if (in_frustum_ret_ref_host[i]) {
            EXPECT_TRUE(float_eq(p_view_ref_host[i], p_view_host[i], 1e-4));
            EXPECT_TRUE(float_eq(p_view_ref_grad_host[i], p_view_grad_host[i], 1e-4));
        }
    }

    free_device(orig_points_device);
    free_device(viewmatrix_device);
    free_device(projmatrix_device);
    free_device(viewmatrix_grad_device);
    free_device(projmatrix_grad_device);
    free_device(p_view_ref_device);
    free_device(p_view_ref_grad_device);
    free_device(in_frustum_ret_ref_device);
    free_device(p_view_device);
    free_device(p_view_grad_device);
    free_device(in_frustum_ret_device);
}
