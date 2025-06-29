#include "auxiliary.h"

#include <gtest/gtest.h>
#include <iostream>
#include <tuple>
#include <utility>
#include "test_utils.h"
#include "float_grad.h"
#include "helper_math.h"

__global__
void call_transform(const float3* p, const float* matrix, float4* transformed) {
    float4 transformed_temp = transformPoint4x4(*p, matrix);
    *transformed = transformed_temp;
}

__global__
void call_transform_jvp(float3* p, 
                        float* matrix, 
                        float4* transformed, 
                        float3* p_grad, 
                        float* matrix_grad, 
                        float4* transformed_grad) {

    float4 transformed_temp = transformPoint4x4(*p, matrix);
    *transformed = transformed_temp;

    *transformed_grad = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float eps = 1e-2f;

    float* aligned_args[19];
    float* aligned_grad[19];
    std::pair<float*, int> args_in[2] = {
        {reinterpret_cast<float*>(p), 3}, 
        {matrix, 16}
    };
    std::pair<float*, int> grad_in[2] = {
        {reinterpret_cast<float*>(p_grad), 3}, 
        {matrix_grad, 16}
    };
    align_params(args_in, 2, aligned_args);
    align_params(grad_in, 2, aligned_grad);

    for (int i = 0; i < 19; i++) {
        float* arg = aligned_args[i];
        float* arg_grad = aligned_grad[i];
        float arg_old = *arg;
        *arg += eps;
        float4 transformed_plus = transformPoint4x4(*p, matrix);
        *arg = arg_old - eps;
        float4 transformed_minus = transformPoint4x4(*p, matrix);
        *transformed_grad += *arg_grad * (transformed_plus - transformed_minus) / (2.0f * eps);
        *arg = arg_old;
    }

}

__global__ 
void call_transform_floatgrad(const float3* p_data, 
                              const float* matrix_data, 
                              float4* transformed_data, 
                              const float3* p_grad,
                              const float* matrix_grad,
                              float4* transformed_grad) {

    FloatGradRef<const float3> p(p_data, p_grad);
    FloatGradArray<const float> matrix(matrix_data, matrix_grad);
    FloatGradRef<float4> transformed(transformed_data, transformed_grad);


    transformed = transformPoint4x4(p, matrix);
}

TEST(ForwardTest, TransformPoint4x4) {
    float3 p_host = {1.0f, 2.0f, 3.0f};
    float3 p_grad_host = {0.1f, 0.2f, 0.3f};
    float* matrix_host = new float[16];
    float* matrix_grad_host = new float[16];

    for (int i = 0; i < 16; ++i) {
        matrix_host[i] = static_cast<float>(i + 1);
        matrix_grad_host[i] = static_cast<float>((i + 1) * 0.2);
    }

    float3* p_device = host_to_device(&p_host, 1);
    float3* p_grad_device = host_to_device(&p_grad_host, 1);
    float* matrix_device = host_to_device(matrix_host, 16);
    float* matrix_grad_device = host_to_device(matrix_grad_host, 16);

    // Launch a 1 thread kernel to test the transformation
    float4* transformed_ref_device;
    float4* transformed_device;
    float4* transformed_grad_device;
    float4* transformed_data_ref_device;
    float4* transformed_grad_ref_device;
    cudaMalloc(&transformed_ref_device, sizeof(float4));
    cudaMalloc(&transformed_device, sizeof(float4));
    cudaMalloc(&transformed_grad_device, sizeof(float4));
    cudaMalloc(&transformed_data_ref_device, sizeof(float4));
    cudaMalloc(&transformed_grad_ref_device, sizeof(float4));

    call_transform<<<1, 1>>>(p_device, 
                             matrix_device, 
                             transformed_ref_device);

    call_transform_floatgrad<<<1, 1>>>(p_device, 
                                       matrix_device, 
                                       transformed_device,
                                       p_grad_device,
                                       matrix_grad_device,
                                       transformed_grad_device);

    call_transform_jvp<<<1, 1>>>(p_device, 
                                 matrix_device, 
                                 transformed_data_ref_device,
                                 p_grad_device,
                                 matrix_grad_device,
                                 transformed_grad_ref_device);
    cudaDeviceSynchronize();

    float4 transformed_ref_host;
    float4 transformed_host;
    float4 transformed_grad_host;
    float* transformed_ref_host_ptr = reinterpret_cast<float*>(&transformed_ref_host);
    float* transformed_host_ptr = reinterpret_cast<float*>(&transformed_host);
    float* transformed_grad_host_ptr = reinterpret_cast<float*>(&transformed_grad_host);
    cudaMemcpy(&transformed_ref_host, transformed_ref_device, sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(&transformed_host, transformed_device, sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(&transformed_grad_host, transformed_grad_device, sizeof(float4), cudaMemcpyDeviceToHost);


    float4 transformed_data_ref_host;
    float4 transformed_grad_ref_host;
    float* transformed_data_ref_host_ptr = reinterpret_cast<float*>(&transformed_data_ref_host);
    float* transformed_grad_ref_host_ptr = reinterpret_cast<float*>(&transformed_grad_ref_host);
    cudaMemcpy(&transformed_data_ref_host, transformed_data_ref_device, sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(&transformed_grad_ref_host, transformed_grad_ref_device, sizeof(float4), cudaMemcpyDeviceToHost);


    EXPECT_TRUE(float_eq(transformed_ref_host, transformed_host));
    EXPECT_TRUE(float_eq(transformed_ref_host, transformed_data_ref_host));
    EXPECT_TRUE(float_eq(transformed_grad_ref_host, transformed_grad_host, 1e-2)) 
        << "Transformed Gradients do not match: "
        << transformed_grad_ref_host_ptr[0] << ", "
        << transformed_grad_ref_host_ptr[1] << ", "
        << transformed_grad_ref_host_ptr[2] << ", "
        << transformed_grad_ref_host_ptr[3] << " vs "
        << transformed_grad_host_ptr[0] << ", "
        << transformed_grad_host_ptr[1] << ", "
        << transformed_grad_host_ptr[2] << ", "
        << transformed_grad_host_ptr[3];

    // Deallocate memory
    cudaFree(p_device);
    cudaFree(matrix_device);
    cudaFree(transformed_ref_device);
    cudaFree(transformed_data_ref_device);
    cudaFree(transformed_grad_ref_device);


    delete[] matrix_host;
    delete[] matrix_grad_host;

}
