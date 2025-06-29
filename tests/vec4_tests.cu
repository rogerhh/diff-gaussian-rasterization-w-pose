#include <cuda.h>
#include <gtest/gtest.h>
#include <iostream>
#include "test_utils.h"
#include "float_grad_helper_math.h"
#include "float_grad.h"
#include "float_grad_vec3.h"
#include "float_grad_vec4.h"
#include "float_grad_mat3.h"
#include "float_grad_mat4.h"
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"

TEST(FloatGradGlm, Vec4) {
    FloatGrad<glm::vec4> a(1.0f, 2.0f, 3.0f, 4.0f);

    expect_near(a.x, FloatGrad<float>(1.0f, 0.0f));
    expect_near(a.y, FloatGrad<float>(2.0f, 0.0f));
    expect_near(a.z, FloatGrad<float>(3.0f, 0.0f));
    expect_near(a.w, FloatGrad<float>(4.0f, 0.0f));

    FloatGrad<float> b0(5.0f, 1.0f);
    FloatGrad<float> b1(6.0f, 2.0f);
    FloatGrad<float> b2(7.0f, 3.0f);
    FloatGrad<float> b3(8.0f, 4.0f);

    FloatGrad<glm::vec4> b(b0, b1, b2, b3);
    expect_near(b.x, b0);
    expect_near(b.y, b1);
    expect_near(b.z, b2);
    expect_near(b.w, b3);

}

TEST(FloatGradGlm, Mat3) {
    float a_data[9];
    float a_grad[9];

    for (int i = 0; i < 9; ++i) {
        a_data[i] = static_cast<float>(i + 1);
        a_grad[i] = static_cast<float>((i + 1) * 0.1);
    }

    FloatGradArray<float> a_arr(a_data, a_grad);

    FloatGrad<glm::mat3> a(a_arr[0],  a_arr[1],  a_arr[2],
                           a_arr[3],  a_arr[4],  a_arr[5],
                           a_arr[6],  a_arr[7],  a_arr[8]);

    for (int i = 0; i < 9; ++i) {
        expect_near(a[i / 3][i % 3], a_arr[i]);
    }
    
}

TEST(FloatGradGlm, Mat4) {
    float a_data[16];
    float a_grad[16];

    for (int i = 0; i < 16; ++i) {
        a_data[i] = static_cast<float>(i + 1);
        a_grad[i] = static_cast<float>((i + 1) * 0.2);
    }

    FloatGradArray<float> a_arr(a_data, a_grad);

    FloatGrad<glm::mat4> a(a_arr[0],  a_arr[1],  a_arr[2],  a_arr[3],
                           a_arr[4],  a_arr[5],  a_arr[6],  a_arr[7],
                           a_arr[8],  a_arr[9],  a_arr[10], a_arr[11],
                           a_arr[12], a_arr[13], a_arr[14], a_arr[15]);

    for (int i = 0; i < 16; ++i) {
        expect_near(a[i / 4][i % 4], a_arr[i]);
    }
    
}

TEST(FloatGradGlm, Length) {
    glm::vec4 v_data(3.0f, 4.0f, 1.0f, 2.0f);
    glm::vec4 v_grad(0.1f, 0.2f, 0.3f, 0.4f);
    FloatGrad<glm::vec4> v(v_data, v_grad);

    auto length = FLOATGRAD::length(v);
}

