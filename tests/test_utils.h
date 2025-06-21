#include "float_grad.h"

#include <string>
#include <vector>

inline bool float_eq(float a, float b, float eps = 1e-6f) {
    return fabs(a - b) < eps;
}

inline bool float_eq(float2 a, float2 b, float eps = 1e-6f) {
    return float_eq(a.x, b.x, eps) && float_eq(a.y, b.y, eps);
}

inline bool float_eq(float3 a, float3 b, float eps = 1e-6f) {
    return float_eq(a.x, b.x, eps) && float_eq(a.y, b.y, eps) && float_eq(a.z, b.z, eps);
}

inline bool float_eq(float4 a, float4 b, float eps = 1e-6f) {
    return float_eq(a.x, b.x, eps) && float_eq(a.y, b.y, eps) &&
           float_eq(a.z, b.z, eps) && float_eq(a.w, b.w, eps);
}

template <typename T1, typename T2,
          typename = std::enable_if_t<is_float_grad<T1>::value
                                      && is_float_grad<T2>::value>>
inline bool float_eq(const T1& a, const T2& b, float eps = 1e-6f) {
    return float_eq(a.data(), b.data(), eps) && float_eq(a.grad(), b.grad(), eps);
}

inline bool expect_near(float a, float b, float eqs = 1e-6f) {
    if (!float_eq(a, b, eqs)) {
        std::cerr << "Expected: " << a << ", but got: " << b << std::endl;
        return false;
    }
    return true;
}

inline bool expect_near(float2 a, float2 b, float eqs = 1e-6f) {
    bool b1 = expect_near(a.x, b.x, eqs);
    bool b2 = expect_near(a.y, b.y, eqs);
    return b1 && b2;
}

inline bool expect_near(float3 a, float3 b, float eqs = 1e-6f) {
    bool b1 = expect_near(a.x, b.x, eqs);
    bool b2 = expect_near(a.y, b.y, eqs);
    bool b3 = expect_near(a.z, b.z, eqs);
    return b1 && b2 && b3;
}

inline bool expect_near(float4 a, float4 b, float eqs = 1e-6f) {
    bool b1 = expect_near(a.x, b.x, eqs);
    bool b2 = expect_near(a.y, b.y, eqs);
    bool b3 = expect_near(a.z, b.z, eqs);
    bool b4 = expect_near(a.w, b.w, eqs);
    return b1 && b2 && b3 && b4;
}

template <typename T1, typename T2,
          typename = std::enable_if_t<is_float_grad<T1>::value
                                      && is_float_grad<T2>::value>>
inline bool expect_near(const T1& a, const T2& b, float eqs = 1e-6f) {
    return expect_near(a.data(), b.data(), eqs) && 
           expect_near(a.grad(), b.grad(), eqs);
}


// Align args in the format of arg1, len1, arg2, len2, ...
__host__ __device__
inline void align_params(std::pair<float*, int>* args, 
                  int len,
                  float** aligned_args) {
    int c = 0;
    for (int i = 0; i < len; i++) {
        float* args_ptr = args[i].first;
        int args_len = args[i].second;
        for (int j = 0; j < args_len; j++) {
            aligned_args[c++] = &args_ptr[j];
        }
    }
}

template <typename T>
T* host_to_device(const T* ptr_host, size_t len) {
    T* ptr_device = nullptr;
    cudaError_t err;

    // Allocate memory on the device
    err = cudaMalloc((void**)&ptr_device, len * sizeof(T));
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
    }

    // Copy data from host to device
    err = cudaMemcpy(ptr_device, ptr_host, len * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(ptr_device); // clean up
        throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }

    return ptr_device;
}

inline void free_device(void* ptr_device) {
    if (ptr_device != nullptr) {
        cudaError_t err = cudaFree(ptr_device);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaFree failed: " + std::string(cudaGetErrorString(err)));
        }
    }
}

// Read a csv to load a 2D array but store it as a contiguous vector of floats.
void read_csv(const std::string& filepath, 
              std::vector<float>& data,
              int& rows,
              int& cols);
