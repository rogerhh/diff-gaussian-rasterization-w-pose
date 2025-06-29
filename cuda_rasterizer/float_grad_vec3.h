#ifndef FLOAT_GRAD_VEC3_H
#define FLOAT_GRAD_VEC3_H

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

// Specialization for get_grad
template <>
inline __host__ __device__
decltype(auto) get_grad<glm::vec3>(const glm::vec3& v) {
    return glm::vec3{0.0f, 0.0f, 0.0f};
}
template <>
inline __host__ __device__
decltype(auto) get_grad<const glm::vec3>(const glm::vec3& v) {
    return glm::vec3{0.0f, 0.0f, 0.0f};
}

template <typename T>
using is_vec3_type = std::is_same<std::decay_t<decltype(get_data(std::declval<T>()))>, glm::vec3>;

template <>
struct FloatGradRef<glm::vec3> : FloatGradRefBase<glm::vec3> {
    // All constructors
    template <typename... Args>
    __host__ __device__
    FloatGradRef(Args&&... args)
    : FloatGradRefBase<glm::vec3>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y),
      z(&data().z, &grad().z) {}

    // All assignment operators
    template <typename OtherType>
    __host__ __device__
    FloatGradRef& operator=(const OtherType& other) {
        FloatGradRefBase<glm::vec3>::operator=(other);
        return *this;
    }

    __host__ __device__
    FloatGradRef<float> operator[](int index) {
        return FloatGradRef<float>(&data()[index], &grad()[index]);
    }

    __host__ __device__
    FloatGradRef<const float> operator[](int index) const {
        return FloatGradRef<const float>(&data()[index], &grad()[index]);
    }

    FloatGradRef<float> x;
    FloatGradRef<float> y;
    FloatGradRef<float> z;
};

template <>
struct FloatGrad<glm::vec3> : FloatGradBase<glm::vec3> {
    // All constructors
    template <typename... Args>
    __host__ __device__
    FloatGrad(Args&&... args)
    : FloatGradBase<glm::vec3>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y),
      z(&data().z, &grad().z) {}

    // All assignment operators
    template <typename OtherType>
    __host__ __device__
    FloatGrad& operator=(const OtherType& other) {
        FloatGradBase<glm::vec3>::operator=(other);
        return *this;
    }

    __host__ __device__
    FloatGradRef<float> operator[](int index) {
        return FloatGradRef<float>(&data()[index], &grad()[index]);
    }

    __host__ __device__
    FloatGradRef<const float> operator[](int index) const {
        return FloatGradRef<const float>(&data()[index], &grad()[index]);
    }

    FloatGradRef<float> x;
    FloatGradRef<float> y;
    FloatGradRef<float> z;
};

template <>
struct FloatGradRef<const glm::vec3> : FloatGradRefBase<const glm::vec3> {
    template <typename... Args>
    __host__ __device__
    FloatGradRef(Args&&... args)
    : FloatGradRefBase<const glm::vec3>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y),
      z(&data().z, &grad().z) {}

    __host__ __device__
    FloatGradRef<const float> operator[](int index) const {
        return FloatGradRef<const float>(&data()[index], &grad()[index]);
    }

    FloatGradRef<const float> x;
    FloatGradRef<const float> y;
    FloatGradRef<const float> z;
};

template <>
struct FloatGrad<const glm::vec3> : FloatGradBase<const glm::vec3> {
    template <typename... Args>
    __host__ __device__
    FloatGrad(Args&&... args)
    : FloatGradBase<const glm::vec3>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y),
      z(&data().z, &grad().z) {}

    __host__ __device__
    FloatGradRef<const float> operator[](int index) const {
        return FloatGradRef<const float>(&data()[index], &grad()[index]);
    }

    FloatGradRef<const float> x;
    FloatGradRef<const float> y;
    FloatGradRef<const float> z;
};

#endif // FLOAT_GRAD_VEC3_H
