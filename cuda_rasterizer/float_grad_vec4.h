#ifndef FLOAT_GRAD_VEC4_H
#define FLOAT_GRAD_VEC4_H

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

template <typename T>
using is_vec4_type = std::is_same<std::decay_t<decltype(get_data(std::declval<T>()))>, glm::vec4>;

template <>
struct FloatGradRef<glm::vec4> : FloatGradRefBase<glm::vec4> {
    // All constructors
    template <typename... Args>
    __host__ __device__
    FloatGradRef(Args&&... args)
    : FloatGradRefBase<glm::vec4>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y),
      z(&data().z, &grad().z),
      w(&data().w, &grad().w) {}

    // All assignment operators
    template <typename OtherType>
    __host__ __device__
    FloatGradRef& operator=(const OtherType& other) {
        FloatGradRefBase<glm::vec4>::operator=(other);
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
    FloatGradRef<float> w;
};

template <>
struct FloatGrad<glm::vec4> : FloatGradBase<glm::vec4> {
    // All constructors
    template <typename... Args>
    __host__ __device__
    FloatGrad(Args&&... args)
    : FloatGradBase<glm::vec4>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y),
      z(&data().z, &grad().z),
      w(&data().w, &grad().w) {}

    // All assignment operators
    template <typename OtherType>
    __host__ __device__
    FloatGrad& operator=(const OtherType& other) {
        FloatGradBase<glm::vec4>::operator=(other);
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
    FloatGradRef<float> w;
};

template <>
struct FloatGradRef<const glm::vec4> : FloatGradRefBase<const glm::vec4> {
    template <typename... Args>
    __host__ __device__
    FloatGradRef(Args&&... args)
    : FloatGradRefBase<const glm::vec4>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y),
      z(&data().z, &grad().z),
      w(&data().w, &grad().w) {}

    __host__ __device__
    FloatGradRef<const float> operator[](int index) const {
        return FloatGradRef<const float>(&data()[index], &grad()[index]);
    }

    FloatGradRef<const float> x;
    FloatGradRef<const float> y;
    FloatGradRef<const float> z;
    FloatGradRef<const float> w;
};

template <>
struct FloatGrad<const glm::vec4> : FloatGradBase<const glm::vec4> {
    template <typename... Args>
    __host__ __device__
    FloatGrad(Args&&... args)
    : FloatGradBase<const glm::vec4>(std::forward<Args>(args)...),
      x(&data().x, &grad().x), 
      y(&data().y, &grad().y),
      z(&data().z, &grad().z),
      w(&data().w, &grad().w) {}

    __host__ __device__
    FloatGradRef<const float> operator[](int index) const {
        return FloatGradRef<const float>(&data()[index], &grad()[index]);
    }

    FloatGradRef<const float> x;
    FloatGradRef<const float> y;
    FloatGradRef<const float> z;
    FloatGradRef<const float> w;
};

#endif // FLOAT_GRAD_VEC4_H
