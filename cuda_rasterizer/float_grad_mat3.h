#ifndef FLOAT_GRAD_MAT3_H
#define FLOAT_GRAD_MAT3_H

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

// Specialization for get_grad
template <>
inline __host__ __device__
decltype(auto) get_grad<glm::mat3>(const glm::mat3& v) {
    return glm::mat3{0.0f, 0.0f, 0.0f, 
                     0.0f, 0.0f, 0.0f, 
                     0.0f, 0.0f, 0.0f};
}
template <>
inline __host__ __device__
decltype(auto) get_grad<const glm::mat3>(const glm::mat3& v) {
    return glm::mat3{0.0f, 0.0f, 0.0f, 
                     0.0f, 0.0f, 0.0f, 
                     0.0f, 0.0f, 0.0f};
}

template <typename T>
using is_mat3_type = std::is_same<std::decay_t<decltype(get_data(std::declval<T>()))>, glm::mat3>;

template <>
struct FloatGradRef<glm::mat3> : FloatGradRefBase<glm::mat3> {
    // All constructors
    template <typename... Args>
    __host__ __device__
    FloatGradRef(Args&&... args)
    : FloatGradRefBase<glm::mat3>(std::forward<Args>(args)...) {}

    // All assignment operators
    template <typename OtherType>
    __host__ __device__
    FloatGradRef& operator=(const OtherType& other) {
        FloatGradRefBase<glm::mat3>::operator=(other);
        return *this;
    }

    __host__ __device__
    FloatGradRef<glm::vec3> operator[](int index) {
        return FloatGradRef<glm::vec3>(&data()[index], &grad()[index]);
    }

    __host__ __device__
    FloatGradRef<const glm::vec3> operator[](int index) const {
        return FloatGradRef<const glm::vec3>(&data()[index], &grad()[index]);
    }
};

template <>
struct FloatGrad<glm::mat3> : FloatGradBase<glm::mat3> {
    // All constructors
    template <typename... Args>
    __host__ __device__
    FloatGrad(Args&&... args)
    : FloatGradBase<glm::mat3>(std::forward<Args>(args)...) {}

    // All assignment operators
    template <typename OtherType>
    __host__ __device__
    FloatGrad& operator=(const OtherType& other) {
        FloatGradBase<glm::mat3>::operator=(other);
        return *this;
    }

    __host__ __device__
    FloatGradRef<glm::vec3> operator[](int index) {
        return FloatGradRef<glm::vec3>(&data()[index], &grad()[index]);
    }

    __host__ __device__
    FloatGradRef<const glm::vec3> operator[](int index) const {
        return FloatGradRef<const glm::vec3>(&data()[index], &grad()[index]);
    }
};

template <>
struct FloatGradRef<const glm::mat3> : FloatGradRefBase<const glm::mat3> {
    template <typename... Args>
    __host__ __device__
    FloatGradRef(Args&&... args)
    : FloatGradRefBase<const glm::mat3>(std::forward<Args>(args)...) {}

    __host__ __device__
    FloatGradRef<const glm::vec3> operator[](int index) const {
        return FloatGradRef<const glm::vec3>(&data()[index], &grad()[index]);
    }
};

template <>
struct FloatGrad<const glm::mat3> : FloatGradBase<const glm::mat3> {
    template <typename... Args>
    __host__ __device__
    FloatGrad(Args&&... args)
    : FloatGradBase<const glm::mat3>(std::forward<Args>(args)...) {}

    __host__ __device__
    FloatGradRef<const glm::vec3> operator[](int index) const {
        return FloatGradRef<const glm::vec3>(&data()[index], &grad()[index]);
    }
};

#endif // FLOAT_GRAD_MAT3_H
