#ifndef FORWARD_IMPL_H
#define FORWARD_IMPL_H

#include "auxiliary.h"
#include "helper_math.h"
#include "math.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <tuple>
#include <float_grad.h>
namespace cg = cooperative_groups;

namespace FORWARD
{

// Perform initial steps for each Gaussian prior to rasterization.
template<int C, typename... JvpArgs>
__global__ void preprocessCUDAJvp(int P, int D, int M,
    bool* clamped,
    const int W, int H,
    const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y,
    int* radii,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered,
    JvpArgs&&... jvp_args)
    // const float* orig_points,
    // const glm::vec3* scales,
    // const float scale_modifier,
    // const glm::vec4* rotations,
    // const float* opacities,
    // const float* shs,
    // const float* cov3D_precomp,
    // const float* colors_precomp,
    // const float* viewmatrix,
    // const float* projmatrix,
    // const glm::vec3* cam_pos,
    // float2* points_xy_image,
    // float* depths,
    // float* cov3Ds,
    // float* rgb,
    // float4* conic_opacity,
{
    // Unpack JVP arguments
    auto jvp_args_tuple = std::forward_as_tuple(std::forward<JvpArgs>(jvp_args)...);
    auto orig_points = std::get<0>(jvp_args_tuple);
    auto scales = std::get<1>(jvp_args_tuple);
    auto scale_modifier = std::get<2>(jvp_args_tuple);
    auto rotations = std::get<3>(jvp_args_tuple);
    auto opacities = std::get<4>(jvp_args_tuple);
    auto shs = std::get<5>(jvp_args_tuple);
    auto cov3D_precomp = std::get<6>(jvp_args_tuple);
    auto colors_precomp = std::get<7>(jvp_args_tuple);
    auto viewmatrix = std::get<8>(jvp_args_tuple);
    auto projmatrix = std::get<9>(jvp_args_tuple);
    auto cam_pos = std::get<10>(jvp_args_tuple);
    auto points_xy_image = std::get<11>(jvp_args_tuple);
    auto depths = std::get<12>(jvp_args_tuple);
    auto cov3Ds = std::get<13>(jvp_args_tuple);
    auto rgb = std::get<14>(jvp_args_tuple);
    auto conic_opacity = std::get<15>(jvp_args_tuple);

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
void preprocessJvp(int P, int D, int M,
    bool* clamped,
    const int W, int H,
    int* radii,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered,
    JvpArgs&&... jvp_args)
    // const float* orig_points,
    // const glm::vec3* scales,
    // const float scale_modifier,
    // const glm::vec4* rotations,
    // const float* opacities,
    // const float* shs,
    // const float* cov3D_precomp,
    // const float* colors_precomp,
    // const float* viewmatrix,
    // const float* projmatrix,
    // const glm::vec3* cam_pos,
    // const float focal_x, float focal_y,
    // const float tan_fovx, float tan_fovy,
    // float2* points_xy_image,
    // float* depths,
    // float* cov3Ds,
    // float* colors,
    // float4* conic_opacity,
{
    preprocessCUDAJvp<CudaRasterizer::NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
        P, D, M,
        clamped,
        W, H,
        radii,
        grid,
        tiles_touched,
        prefiltered,
        std::forward<JvpArgs>(jvp_args)...
        );
}

} // namespace FORWARD

#endif // FORWARD_IMPL_H
