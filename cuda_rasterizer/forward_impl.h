#ifndef FORWARD_IMPL_H
#define FORWARD_IMPL_H

#include "auxiliary.h"
#include "helper_math.h"
#include "math.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <tuple>
#include <float_grad.h>
#include "float_grad_vec4.h"
#include "float_grad_mat4.h"
namespace cg = cooperative_groups;

namespace FORWARD
{

// Forward version of 2D covariance matrix computation
template <typename... JvpArgs>
__device__ float3 computeCov2D(JvpArgs&&... jvp_args)
            // const float3& mean, float focal_x, float focal_y, 
            // float tan_fovx, float tan_fovy, const float* cov3D, 
            // const float* viewmatrix
{
    auto jvp_args_tuple = std::forward_as_tuple(std::forward<JvpArgs>(jvp_args)...);
    auto mean = std::get<0>(jvp_args_tuple);
    auto focal_x = std::get<1>(jvp_args_tuple);
    auto focal_y = std::get<2>(jvp_args_tuple);
    auto tan_fovx = std::get<3>(jvp_args_tuple);
    auto tan_fovy = std::get<4>(jvp_args_tuple);
    auto cov3D = std::get<5>(jvp_args_tuple);
    auto viewmatrix = std::get<6>(jvp_args_tuple);

    // The following models the steps outlined by equations 29
    // and 31 in "EWA Splatting" (Zwicker et al., 2002). 
    // Additionally considers aspect / scaling of viewport.
    // Transposes used to account for row-/column-major conventions.
    auto t = transformPoint4x3(mean, viewmatrix);

    const auto limx = 1.3f * tan_fovx;
    const auto limy = 1.3f * tan_fovy;
    const auto txtz = t.x / t.z;
    const auto tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);

    glm::mat3 W = glm::mat3(
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    glm::mat3 T = W * J;

    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

template <typename T1, typename T2, typename T3, typename T4>
__device__ void computeCov3D(T1 scale, T2 mod, T3 rot, T4 cov3D)
{
    using Stype = std::conditional_t<is_float_grad<T1>::value || is_float_grad<T2>::value, 
                                     FloatGrad<glm::mat3>, glm::mat3>;

    // Create scaling matrix
    Stype S = Stype(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    using Qtype = std::conditional_t<is_float_grad<T3>::value, FloatGrad<glm::vec4>, glm::vec4>;

    // Normalize quaternion to get valid rotation
    Qtype q = rot;// / glm::length(rot);
    auto r = q.x;
    auto x = q.y;
    auto y = q.z;
    auto z = q.w;

    using Rtype = std::conditional_t<is_float_grad<T3>::value, FloatGrad<glm::mat3>, glm::mat3>;

    // Compute rotation matrix from quaternion
    Rtype R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    using Mtype = std::conditional_t<is_float_grad<Stype>::value || is_float_grad<Rtype>::value, 
                                     FloatGrad<glm::mat3>, glm::mat3>;

    Mtype M = S * R;

    // Compute 3D world covariance matrix Sigma
    Mtype Sigma = glm::transpose(M) * M;

    // Covariance is symmetric, only store upper right
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C, typename... JvpArgs>
__global__ void preprocessCUDAJvp(JvpArgs&&... jvp_args)
    // int P, int D, int M,
    // const float* orig_points,
    // const glm::vec3* scales,
    // const float scale_modifier,
    // const glm::vec4* rotations,
    // const float* opacities,
    // const float* shs,
    // bool* clamped,
    // const float* cov3D_precomp,
    // const float* colors_precomp,
    // const float* viewmatrix,
    // const float* projmatrix,
    // const glm::vec3* cam_pos,
    // const int W, int H,
    // const float focal_x, float focal_y,
    // const float tan_fovx, float tan_fovy,
    // int* radii,
    // float2* points_xy_image,
    // float* depths,
    // float* cov3Ds,
    // float* colors,
    // float4* conic_opacity,
    // const dim3 grid,
    // uint32_t* tiles_touched,
    // bool prefiltered
{
    // Unpack JVP arguments
    auto jvp_args_tuple = std::forward_as_tuple(std::forward<JvpArgs>(jvp_args)...);
    int P = std::get<0>(jvp_args_tuple);
    int D = std::get<1>(jvp_args_tuple);
    int M = std::get<2>(jvp_args_tuple);
    auto orig_points = std::get<3>(jvp_args_tuple);
    auto scales = std::get<4>(jvp_args_tuple);
    auto scale_modifier = std::get<5>(jvp_args_tuple);
    auto rotations = std::get<6>(jvp_args_tuple);
    auto opacities = std::get<7>(jvp_args_tuple);
    auto shs = std::get<8>(jvp_args_tuple);
    bool* clamped = std::get<9>(jvp_args_tuple);
    auto cov3D_precomp = std::get<10>(jvp_args_tuple);
    auto colors_precomp = std::get<11>(jvp_args_tuple);
    auto viewmatrix = std::get<12>(jvp_args_tuple);
    auto projmatrix = std::get<13>(jvp_args_tuple);
    auto cam_pos = std::get<14>(jvp_args_tuple);
    int W = std::get<15>(jvp_args_tuple);
    int H = std::get<16>(jvp_args_tuple);
    auto focal_x = std::get<17>(jvp_args_tuple);
    auto focal_y = std::get<18>(jvp_args_tuple);
    auto tan_fovx = std::get<19>(jvp_args_tuple);
    auto tan_fovy = std::get<20>(jvp_args_tuple);
    int* radii = std::get<21>(jvp_args_tuple);
    auto points_xy_image = std::get<22>(jvp_args_tuple);
    auto depths = std::get<23>(jvp_args_tuple);
    auto cov3Ds = std::get<24>(jvp_args_tuple);
    auto colors = std::get<25>(jvp_args_tuple);
    auto conic_opacity = std::get<26>(jvp_args_tuple);
    const dim3 grid = std::get<27>(jvp_args_tuple);
    uint32_t* tiles_touched = std::get<28>(jvp_args_tuple);
    bool prefiltered = std::get<29>(jvp_args_tuple);

    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Initialize radius and touched tiles to 0. If this isn't changed,
    // this Gaussian will not be processed further.
    radii[idx] = 0;
    tiles_touched[idx] = 0;

    // Perform near culling, quit if outside.
    FloatGrad<float3> p_view(make_float3(0));
    if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
        return;

    // Transform point by projecting
    auto p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
    auto p_hom = transformPoint4x4(p_orig, projmatrix);
    auto p_w = 1.0f / (p_hom.w + 0.0000001f);
    auto p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    // If 3D covariance matrix is precomputed, use it, otherwise compute
    // from scaling and rotation parameters. 
    const float* cov3D;
    if (cov3D_precomp != nullptr)
    {
        cov3D = cov3D_precomp + idx * 6;
    }
    else
    {
        computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
        cov3D = cov3Ds + idx * 6;
    }

    // Compute 2D screen-space covariance matrix
    auto cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

    // // Invert covariance (EWA algorithm)
    // float det = (cov.x * cov.z - cov.y * cov.y);
    // if (det == 0.0f)
    //     return;
    // float det_inv = 1.f / det;
    // float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

    // // Compute extent in screen space (by finding eigenvalues of
    // // 2D covariance matrix). Use extent to compute a bounding rectangle
    // // of screen-space tiles that this Gaussian overlaps with. Quit if
    // // rectangle covers 0 tiles. 
    // float mid = 0.5f * (cov.x + cov.z);
    // float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    // float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    // float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
    // float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
    // uint2 rect_min, rect_max;
    // getRect(point_image, my_radius, rect_min, rect_max, grid);
    // if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
    //     return;

    // // If colors have been precomputed, use them, otherwise convert
    // // spherical harmonics coefficients to RGB color.
    // if (colors_precomp == nullptr)
    // {
    //     glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
    //     rgb[idx * C + 0] = result.x;
    //     rgb[idx * C + 1] = result.y;
    //     rgb[idx * C + 2] = result.z;
    // }

    // // Store some useful helper data for the next steps.
    // depths[idx] = p_view.z;
    // radii[idx] = my_radius;
    // points_xy_image[idx] = point_image;
    // // Inverse 2D covariance and opacity neatly pack into one float4
    // conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
    // tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

template <typename... JvpArgs>
void preprocessJvp(JvpArgs&&... jvp_args)
    // int P, int D, int M,
    // const float* orig_points,
    // const glm::vec3* scales,
    // const float scale_modifier,
    // const glm::vec4* rotations,
    // const float* opacities,
    // const float* shs,
    // bool* clamped,
    // const float* cov3D_precomp,
    // const float* colors_precomp,
    // const float* viewmatrix,
    // const float* projmatrix,
    // const glm::vec3* cam_pos,
    // const int W, int H,
    // const float focal_x, float focal_y,
    // const float tan_fovx, float tan_fovy,
    // int* radii,
    // float2* points_xy_image,
    // float* depths,
    // float* cov3Ds,
    // float* colors,
    // float4* conic_opacity,
    // const dim3 grid,
    // uint32_t* tiles_touched,
    // bool prefiltered
{
    auto jvp_args_tuple = std::forward_as_tuple(std::forward<JvpArgs>(jvp_args)...);
    int P = std::get<0>(jvp_args_tuple);

    preprocessCUDAJvp<CudaRasterizer::NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
        std::forward<JvpArgs>(jvp_args)...
    );
}

} // namespace FORWARD

#endif // FORWARD_IMPL_H
