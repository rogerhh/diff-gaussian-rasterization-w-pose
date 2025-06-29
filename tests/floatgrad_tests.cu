#include <gtest/gtest.h>
#include <iostream>
#include "test_utils.h"
#include "float_grad.h"

TEST(FloatGrad, Float2) {
    // Scalar operations still need to work
    float2 a_data{-1.0f, 2.0f};
    float2 a_grad{0.1f, 0.2f};

    FloatGrad<float2> a(a_data, a_grad);

    FloatGrad<float2> b = fabs(a);

}
