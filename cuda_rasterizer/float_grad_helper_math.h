#ifndef FLOAT_GRAD_HELPER_MATH_H
#define FLOAT_GRAD_HELPER_MATH_H

#include <cuda.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

// Overload some helpful namespace types and functions for floatgrad types
#include "float_grad_vec3.h"
#include "float_grad_vec4.h"
#include "float_grad_mat3.h"
#include "float_grad_mat4.h"

#include "helper_math.h"

namespace FLOATGRAD {

// We don't want to overload functions in the glm namespace so we have a dispatcher

template <typename T>
struct always_false_type : std::false_type {};

template <typename T>
inline __host__ __device__
void print_type() {
    static_assert(always_false_type<T>::value, "Unsupported type for print_type");
}

template <typename T1, typename T2>
inline __host__ __device__
auto dot(const T1& a, const T2& b) {
    if constexpr (!(is_float_grad<T1>::value || is_float_grad<T2>::value)) {
        return glm::dot(a, b);
    }
    else {
        return FloatGrad<float>(glm::dot(get_data(a), get_data(b)),
                                glm::dot(get_data(a), get_grad(b))
                                    + glm::dot(get_grad(a), get_data(b)));
    }
}

template <typename T>
inline __host__ __device__
auto length(const T& v) {
    if constexpr (!is_float_grad<T>::value) {
        return glm::length(v);
    }
    else {
        auto d = FLOATGRAD::dot(v, v);
        return sqrtf(d);
    }
}

template <typename T1, typename T2>
inline __host__ __device__
auto max(T1 a, T2 b) {
    if constexpr (!(is_float_grad<T1>::value || is_float_grad<T2>::value)) {
        return glm::max(a, b);
    }
    else if constexpr (is_vec3_type<T1>::value) {
        if constexpr (is_float_type<T2>::value) {
            return FloatGrad<glm::vec3>(fmaxf(a.x, b), fmaxf(a.y, b), fmaxf(a.z, b));
        }
        else if constexpr (is_vec3_type<T2>::value) {
            return FloatGrad<glm::vec3>(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
        }
        else {
            static_assert(always_false<T1>::value, "Unsupported type combination for max");
            static_assert(always_false<T2>::value, "Unsupported type combination for max");
        }
    }
    else if constexpr (is_vec4_type<T1>::value) {
        if constexpr (is_float_type<T2>::value) {
            return FloatGrad<glm::vec4>(fmaxf(a.x, b), fmaxf(a.y, b), 
                                        fmaxf(a.z, b), fmaxf(a.w, b));
        }
        else if constexpr (is_vec4_type<T2>::value) {
            return FloatGrad<glm::vec4>(fmaxf(a.x, b.x), fmaxf(a.y, b.y), 
                                        fmaxf(a.z, b.z), fmaxf(a.w, b.w));
        }
        else {
            static_assert(always_false<T1>::value, "Unsupported type combination for max");
            static_assert(always_false<T2>::value, "Unsupported type combination for max");
        }
    }
}

template <typename T>
inline __host__ __device__
auto transpose(const T& m) {
    if constexpr (!is_float_grad<T>::value) {
        return glm::transpose(m);
    }
    else if constexpr (is_mat3_type<T>::value) {
        return FloatGrad<glm::mat3>(glm::transpose(m.data()), glm::transpose(m.grad()));
    }
    else if constexpr (is_mat4_type<T>::value) {
        return FloatGrad<glm::mat4>(glm::transpose(m.data()), glm::transpose(m.grad()));
    }
    else {
        static_assert(always_false<T>::value, "Unsupported type combination for max");
    }
}

}

#endif // FLOAT_GRAD_HELPER_MATH_H
