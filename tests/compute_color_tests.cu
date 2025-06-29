#include "auxiliary.h"

#include <gtest/gtest.h>
#include <iostream>
#include <tuple>
#include <utility>
#include "test_utils.h"
#include "float_grad.h"
#include "helper_math.h"
#include "forward_impl.h"

#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

__global__
void call_computeColorFromSH_jvp(int idx, int D, int M, 
                                 float3* orig_points, 
                                 glm::vec3* cam_pos, 
                                 float* shs, 
                                 bool* clamped,
                                 glm::vec3* result,
                                 glm::vec3* cam_pos_grad,
                                 float* shs_grad,
                                 glm::vec3* result_grad) {

    glm::vec3* sh = ((glm::vec3*) shs) + idx * M;
    glm::vec3* sh_grad = ((glm::vec3*) shs_grad) + idx * M;

    result_grad[idx] = glm::vec3(0.0f, 0.0f, 0.0f);

    float eps = 1e-2f;

    float* aligned_args[6];
    float* aligned_grad[6];

    std::pair<float*, int> args_in[2] = {
        {(float*) cam_pos, 3},
        {(float*) sh, 3}
    };
    std::pair<float*, int> grad_in[2] = {
        {(float*) cam_pos_grad, 3},
        {(float*) sh_grad, 3}
    };
    align_params(args_in, 2, aligned_args);
    align_params(grad_in, 2, aligned_grad);

    for (int i = 0; i < 6; i++) {
        float* arg = aligned_args[i];
        float* arg_grad = aligned_grad[i];
        float arg_old = *arg;
        *arg += eps;
        glm::vec3 result_plus, result_minus;
        result_plus = FORWARD::computeColorFromSH(idx, D, M, cast<glm::vec3>(orig_points), 
                                                  *cam_pos, shs, clamped);
        *arg = arg_old - eps;
        result_minus = FORWARD::computeColorFromSH(idx, D, M, cast<glm::vec3>(orig_points), 
                                                   *cam_pos, shs, clamped);
        glm::vec3 result_diff = result_plus - result_minus;
        result_diff /= (2.0f * eps);
        result_diff *= *arg_grad;
        result_grad[idx] += result_diff;
        *arg = arg_old;
    }

    result[idx] = FORWARD::computeColorFromSH(idx, D, M, cast<glm::vec3>(orig_points), 
                                              *cam_pos, shs, clamped);
}

__global__
void call_computeColorFromSH_floatgrad(int idx, int D, int M, 
                                       float3* orig_points, 
                                       FloatGradArray<glm::vec3> cam_pos, 
                                       FloatGradArray<float> shs, 
                                       bool* clamped,
                                       FloatGradArray<glm::vec3> result) {
    result[idx] = FORWARD::computeColorFromSH(idx, D, M, cast<glm::vec3>(orig_points), 
                                              cam_pos[0], shs, clamped);
}

TEST(ForwardTest, ComputeColorFromSHTest) {
    std::vector<float> orig_points_host;
    int orig_points_rows, orig_points_cols;
    read_csv("data/means3D.csv", orig_points_host, orig_points_rows, orig_points_cols);
    EXPECT_EQ(orig_points_cols, 3);

    std::vector<int> sh_degree_host;
    int sh_degree_rows, sh_degree_cols;
    read_csv("data/sh_degree.csv", sh_degree_host, sh_degree_rows, sh_degree_cols);

    std::vector<float> shs_host;
    int shs_rows, shs_cols;
    read_csv("data/sh.csv", shs_host, shs_rows, shs_cols);

    std::vector<float> cam_pos_host;
    int cam_pos_rows, cam_pos_cols;
    read_csv("data/campos.csv", cam_pos_host, cam_pos_rows, cam_pos_cols);

    std::vector<float> cam_pos_grad_host;
    for (int i = 0; i < cam_pos_rows * cam_pos_cols; i++) {
        cam_pos_grad_host.push_back((i * 0.2f) + 0.5f);
    }

    std::vector<float> shs_grad_host;
    for (int i = 0; i < shs_rows * shs_cols; i++) {
        shs_grad_host.push_back(((i % 13) * 0.1f) + 0.3f);
    }

    bool* clamped_ref_host = new bool[orig_points_rows * 3];

    float3* orig_points_device = host_to_device((float3*) orig_points_host.data(), orig_points_rows * orig_points_cols);
    int sh_degree = sh_degree_host[0];
    glm::vec3* cam_pos_device = host_to_device((glm::vec3*) cam_pos_host.data(), cam_pos_rows);
    float* shs_device = host_to_device(shs_host.data(), shs_rows * shs_cols);
    bool* clamped_ref_device = host_to_device(clamped_ref_host, orig_points_rows * 3);
    glm::vec3* cam_pos_grad_device = host_to_device((glm::vec3*) cam_pos_grad_host.data(), cam_pos_rows);
    float* shs_grad_device = host_to_device(shs_grad_host.data(), shs_rows * shs_cols);

    std::vector<glm::vec3> result_ref_host(orig_points_rows);
    std::vector<glm::vec3> result_grad_ref_host(orig_points_rows, glm::vec3(0.0f));
    glm::vec3* result_ref_device = host_to_device(result_ref_host.data(), orig_points_rows);
    glm::vec3* result_grad_ref_device = host_to_device(result_grad_ref_host.data(), orig_points_rows);

    call_computeColorFromSH_jvp<<<1, 1>>>(
        0, sh_degree, shs_cols,
        orig_points_device,
        cam_pos_device, 
        shs_device, 
        clamped_ref_device,
        result_ref_device,
        cam_pos_grad_device,
        shs_grad_device,
        result_grad_ref_device
    );

    cudaDeviceSynchronize();

    device_to_host(clamped_ref_host, clamped_ref_device, orig_points_rows * 3);
    device_to_host(result_ref_host.data(), result_ref_device, orig_points_rows);
    device_to_host(result_grad_ref_host.data(), result_grad_ref_device, orig_points_rows);

    // for (int i = 0; i < 1; i++) {
    //     std::cout << "result[" << i << "] = " 
    //               << result_ref_host[i].x << ", "
    //               << result_ref_host[i].y << ", "
    //               << result_ref_host[i].z << std::endl;
    //     std::cout << "result_grad[" << i << "] = "
    //               << result_grad_ref_host[i].x << ", "
    //               << result_grad_ref_host[i].y << ", "
    //               << result_grad_ref_host[i].z << std::endl;
    // }

    bool* clamped_host = new bool[orig_points_rows * 3];
    std::vector<glm::vec3> result_host(orig_points_rows);
    std::vector<glm::vec3> result_grad_host(orig_points_rows, glm::vec3(0.0f));

    bool* clamped_device = host_to_device(clamped_host, orig_points_rows * 3);
    glm::vec3* result_device = host_to_device(result_host.data(), orig_points_rows);
    glm::vec3* result_grad_device = host_to_device(result_grad_host.data(), orig_points_rows);

    FloatGradArray<glm::vec3> cam_pos_floatgrad(cam_pos_device, cam_pos_grad_device);
    FloatGradArray<float> shs_floatgrad(shs_device, shs_grad_device);
    FloatGradArray<glm::vec3> result_floatgrad(result_device, result_grad_device);

    call_computeColorFromSH_floatgrad<<<1, 1>>>(
        0, sh_degree, orig_points_rows,
        orig_points_device,
        cam_pos_floatgrad,
        shs_floatgrad,
        clamped_device,
        result_floatgrad
    );

    cudaDeviceSynchronize();

    device_to_host(clamped_host, clamped_device, orig_points_rows * 3);
    device_to_host(result_host.data(), result_device, orig_points_rows);
    device_to_host(result_grad_host.data(), result_grad_device, orig_points_rows);

    // for (int i = 0; i < 1; i++) {
    //     std::cout << "result[" << i << "] = " 
    //               << result_host[i].x << ", "
    //               << result_host[i].y << ", "
    //               << result_host[i].z << std::endl;
    //     std::cout << "result_grad[" << i << "] = "
    //               << result_grad_host[i].x << ", "
    //               << result_grad_host[i].y << ", "
    //               << result_grad_host[i].z << std::endl;
    // }

    for (int i = 0; i < 1; i++) {
        EXPECT_TRUE(expect_near(result_host[i].x, result_ref_host[i].x, 1e-4f));
        EXPECT_TRUE(expect_near(result_host[i].y, result_ref_host[i].y, 1e-4f));
        EXPECT_TRUE(expect_near(result_host[i].z, result_ref_host[i].z, 1e-4f));
        EXPECT_TRUE(expect_near(result_grad_host[i].x, result_grad_ref_host[i].x, 1e-4f));
        EXPECT_TRUE(expect_near(result_grad_host[i].y, result_grad_ref_host[i].y, 1e-4f));
        EXPECT_TRUE(expect_near(result_grad_host[i].z, result_grad_ref_host[i].z, 1e-4f));
    }

    free(clamped_ref_host);
    free(clamped_host);
    free_device(orig_points_device);
    free_device(cam_pos_device);
    free_device(shs_device);
    free_device(clamped_ref_device);
    free_device(cam_pos_grad_device);
    free_device(shs_grad_device);
    free_device(result_ref_device);
    free_device(result_grad_ref_device);
}
