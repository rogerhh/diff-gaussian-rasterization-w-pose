#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "float_grad.h"

#include <string>
#include <vector>

inline bool float_eq(float a, float b, float eps = 1e-6f, bool rel = false) {
    if(!rel) {
        // Absolute comparison
        return fabs(a - b) < eps;
    }
    else {
        return fabs(a - b) / fabs(a) < eps;
    }
}

inline bool float_eq(float2 a, float2 b, float eps = 1e-6f, bool rel = false) {
    return float_eq(a.x, b.x, eps, rel) && float_eq(a.y, b.y, eps, rel);
}

inline bool float_eq(float3 a, float3 b, float eps = 1e-6f, bool rel = false) {
    return float_eq(a.x, b.x, eps, rel) 
           && float_eq(a.y, b.y, eps, rel) 
           && float_eq(a.z, b.z, eps, rel);
}

inline bool float_eq(float4 a, float4 b, float eps = 1e-6f, bool rel = false) {
    return float_eq(a.x, b.x, eps, rel) && float_eq(a.y, b.y, eps, rel) &&
           float_eq(a.z, b.z, eps, rel) && float_eq(a.w, b.w, eps, rel);
}

template <typename T1, typename T2,
          typename = std::enable_if_t<is_float_grad<T1>::value
                                      && is_float_grad<T2>::value>>
inline bool float_eq(const T1& a, const T2& b, float eps = 1e-6f, 
                     bool rel = false) {
    return float_eq(a.data(), b.data(), eps, rel) && float_eq(a.grad(), b.grad(), eps, rel);
}

inline bool expect_near(float a, float b, float eqs = 1e-6f, 
                        bool rel = false, bool verbose = true) {
    if (!float_eq(a, b, eqs, rel)) {
        float rel_error = fabs((a - b) / a);
        if (verbose) {
            std::cerr << "Expected: " << a << ", but got: " << b 
                        << " (absolute error: " << fabs(a - b)
                        << ") (relative error: " << rel_error << ")" << std::endl;  
        }
        return false;
    }
    return true;
}

inline bool expect_near(float2 a, float2 b, float eqs = 1e-6f, bool rel = false) {
    bool b1 = expect_near(a.x, b.x, eqs, rel);
    bool b2 = expect_near(a.y, b.y, eqs, rel);
    return b1 && b2;
}

inline bool expect_near(float3 a, float3 b, float eqs = 1e-6f, bool rel = false) {
    bool b1 = expect_near(a.x, b.x, eqs, rel);
    bool b2 = expect_near(a.y, b.y, eqs, rel);
    bool b3 = expect_near(a.z, b.z, eqs, rel);
    return b1 && b2 && b3;
}

inline bool expect_near(float4 a, float4 b, float eqs = 1e-6f, bool rel = false) {
    bool b1 = expect_near(a.x, b.x, eqs, rel);
    bool b2 = expect_near(a.y, b.y, eqs, rel);
    bool b3 = expect_near(a.z, b.z, eqs, rel);
    bool b4 = expect_near(a.w, b.w, eqs, rel);
    return b1 && b2 && b3 && b4;
}

template <typename T1, typename T2,
          typename = std::enable_if_t<is_float_grad<T1>::value
                                      && is_float_grad<T2>::value>>
inline bool expect_near(const T1& a, const T2& b, float eps = 1e-6f, 
                     bool rel = false) {
    return expect_near(a.data(), b.data(), eps, rel) 
            && expect_near(a.grad(), b.grad(), eps, rel);
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
    if (ptr_host == nullptr || len == 0) {
        return nullptr;
    }

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

template <typename T>
void device_to_host(T* ptr_host,
                           const T* ptr_device, 
                           size_t len) {
    cudaError_t err = cudaMemcpy(ptr_host, ptr_device, len * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
    }
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
template <typename T>
void read_csv(const std::string& filepath, 
              std::vector<T>& data,
              int& rows,
              int& cols);

template <typename T>
T read_scalar(const std::string& filepath);

__global__
inline void update_arg(float* arg, float eps, float* arg_old, int mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        if (mode == 0) {
            *arg_old = *arg;
            *arg += eps;
        }
        else if (mode == 1) {
            *arg = *arg_old - eps;
        }
        else if (mode == 2) {
            *arg = *arg_old;
        }
    }
}

#include "test_utils_impl.h"

#endif // TEST_UTILS_H
