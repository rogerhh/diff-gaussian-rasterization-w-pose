#include <cuda.h>
#include <gtest/gtest.h>
#include <iostream>
#include <tuple>
#include <utility>
#include "forward_impl.h"
#include "test_utils.h"
#include "float_grad.h"
#include "helper_math.h"
#include "float_grad_helper_math.h"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

// This is the ground truth implementation
__device__ void computeCov3D_local(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
    // Create scaling matrix
    glm::mat3 S = glm::mat3(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    // Normalize quaternion to get valid rotation
    glm::vec4 q = rot;// / glm::length(rot);
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    // Compute rotation matrix from quaternion
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    glm::mat3 M = S * R;

    // Compute 3D world covariance matrix Sigma
    glm::mat3 Sigma = glm::transpose(M) * M;

    // Covariance is symmetric, only store upper right
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

__host__ __device__
void array_plus(int len, float* a, float* b, float* result) {
    for (int i = 0; i < len; i++) {
        result[i] = a[i] + b[i];
    }
}

__host__ __device__
void array_minus(int len, float* a, float* b, float* result) {
    for (int i = 0; i < len; i++) {
        result[i] = a[i] - b[i];
    }
}

__host__ __device__
void array_scale(int len, float* a, float scale, float* result) {
    for (int i = 0; i < len; i++) {
        result[i] = a[i] * scale;
    }
}

__global__
void call_computeCov3D_jvp(glm::vec3 scale, 
                           float mod, 
                           glm::vec4 rot, 
                           float* cov3D,
                           glm::vec3 scale_grad,
                           float mod_grad,
                           glm::vec4 rot_grad,
                           float* cov3D_grad) {

    computeCov3D_local(scale, mod, rot, cov3D);

    for (int i = 0; i < 6; i++) { cov3D_grad[i] = 0.0f; }

    float eps = 1e-2f;

    float* aligned_args[8];
    float* aligned_grad[8];
    std::pair<float*, int> args_in[3] = {
        {reinterpret_cast<float*>(&scale), 3},
        {&mod, 1},
        {reinterpret_cast<float*>(&rot), 4}
    };
    std::pair<float*, int> grad_in[3] = {
        {reinterpret_cast<float*>(&scale_grad), 3},
        {&mod_grad, 1},
        {reinterpret_cast<float*>(&rot_grad), 4}
    };
    align_params(args_in, 3, aligned_args);
    align_params(grad_in, 3, aligned_grad);

    for (int i = 0; i < 8; i++) {
        float* arg = aligned_args[i];
        float* arg_grad = aligned_grad[i];
        float arg_old = *arg;
        *arg += eps;
        float cov3D_plus[6], cov3D_minus[6], cov3D_diff[6];
        computeCov3D_local(scale, mod, rot, cov3D_plus);
        *arg = arg_old - eps;
        computeCov3D_local(scale, mod, rot, cov3D_minus);
        array_minus(6, cov3D_plus, cov3D_minus, cov3D_diff);
        array_scale(6, cov3D_diff, *arg_grad / (2.0f * eps), cov3D_diff);
        array_plus(6, cov3D_grad, cov3D_diff, cov3D_grad);
        *arg = arg_old;
    }

}

__global__
void call_computeCov3D_floatgrad(FloatGrad<glm::vec3> scale,
                                 FloatGrad<float> mod,
                                 FloatGrad<glm::vec4> rot,
                                 FloatGradArray<float> cov3D) {
    FORWARD::computeCov3D(scale, mod, rot, cov3D);
}

TEST(ForwardTest, ComputeCov3DTest) {
    glm::vec3 scale_host = {0.2f, 1.3f, 0.67f};
    float mod_host = 0.5f;
    glm::vec4 rot_host = {0.7f, 0.25f, 0.13f, -0.4f};
    glm::vec3 scale_grad_host = {0.01f, 0.02f, 0.03f};
    float mod_grad_host = 0.1f;
    glm::vec4 rot_grad_host = {0.01f, -0.02f, 0.031f, 0.14f};
    std::vector<float> cov3D_ref_host(6, 0);
    std::vector<float> cov3D_grad_ref_host(6, 0);

    float* cov3D_ref_device = host_to_device(cov3D_ref_host.data(), 6);
    float* cov3D_grad_ref_device = host_to_device(cov3D_grad_ref_host.data(), 6);

    call_computeCov3D_jvp<<<1, 1>>>(scale_host, mod_host, rot_host, 
                                     cov3D_ref_device, 
                                     scale_grad_host, mod_grad_host, rot_grad_host, 
                                     cov3D_grad_ref_device);

    cudaDeviceSynchronize();

    device_to_host(cov3D_ref_host.data(), cov3D_ref_device, 6);
    device_to_host(cov3D_grad_ref_host.data(), cov3D_grad_ref_device, 6);

    // for (int i = 0; i < 6; i++) {
    //     std::cout << "cov3D_ref[" << i << "] = " << cov3D_ref_host[i] << std::endl;
    // }
    // for (int i = 0; i < 6; i++) {
    //     std::cout << "cov3D_grad_ref[" << i << "] = " << cov3D_grad_ref_host[i] << std::endl;
    // }

    std::vector<float> cov3D_host(6, -1.00);
    std::vector<float> cov3D_grad_host(6, 0);

    float* cov3D_device = host_to_device(cov3D_host.data(), 6);
    float* cov3D_grad_device = host_to_device(cov3D_grad_host.data(), 6);

    FloatGrad<glm::vec3> scale_floatgrad(scale_host, scale_grad_host);
    FloatGrad<float> mod_floatgrad(mod_host, mod_grad_host);
    FloatGrad<glm::vec4> rot_floatgrad(rot_host, rot_grad_host);
    FloatGradArray<float> cov3D_device_floatgrad(cov3D_device, cov3D_grad_device);


    call_computeCov3D_floatgrad<<<1, 1>>>(scale_floatgrad, mod_floatgrad, rot_floatgrad, cov3D_device_floatgrad);
    cudaDeviceSynchronize();

    device_to_host(cov3D_host.data(), cov3D_device_floatgrad.data_ptr(), 6);
    device_to_host(cov3D_grad_host.data(), cov3D_device_floatgrad.grad_ptr(), 6);

    // for (int i = 0; i < 6; i++) {
    //     std::cout << "cov3D[" << i << "] = " << cov3D_host[i] << std::endl;
    // }
    // for (int i = 0; i < 6; i++) {
    //     std::cout << "cov3D_grad[" << i << "] = " << cov3D_grad_host[i] << std::endl;
    // }

    for (int i = 0; i < 6; i++) {
        EXPECT_TRUE(expect_near(cov3D_host[i], cov3D_ref_host[i], 1e-4f));
        EXPECT_TRUE(expect_near(cov3D_grad_host[i], cov3D_grad_ref_host[i], 1e-4f));
    }

    free_device(cov3D_ref_device);
    free_device(cov3D_grad_ref_device);
    free_device(cov3D_device);
    free_device(cov3D_grad_device);
}
