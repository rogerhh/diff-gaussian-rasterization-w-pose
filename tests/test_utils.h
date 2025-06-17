#include "float_grad.h"

__host__ __device__
inline bool float_eq(float a, float b, float eps = 1e-6f) {
    return fabs(a - b) < eps;
}

__host__ __device__
inline bool float_eq(float2 a, float2 b, float eps = 1e-6f) {
    return float_eq(a.x, b.x, eps) && float_eq(a.y, b.y, eps);
}

__host__ __device__
inline bool float_eq(float3 a, float3 b, float eps = 1e-6f) {
    return float_eq(a.x, b.x, eps) && float_eq(a.y, b.y, eps) && float_eq(a.z, b.z, eps);
}

__host__ __device__
inline bool float_eq(float4 a, float4 b, float eps = 1e-6f) {
    return float_eq(a.x, b.x, eps) && float_eq(a.y, b.y, eps) &&
           float_eq(a.z, b.z, eps) && float_eq(a.w, b.w, eps);
}

template <typename T1, typename T2,
          typename = std::enable_if_t<is_float_grad<T1>::value
                                      && is_float_grad<T2>::value>>
__host__ __device__
inline bool float_eq(const T1& a, const T2& b, float eps = 1e-6f) {
    return float_eq(a.data(), b.data(), eps) && float_eq(a.grad(), b.grad(), eps);
}
