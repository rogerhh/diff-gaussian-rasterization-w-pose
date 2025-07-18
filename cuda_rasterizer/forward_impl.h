#ifndef FORWARD_IMPL_H
#define FORWARD_IMPL_H

#include "auxiliary.h"
#include "helper_math.h"
#include "float_grad_helper_math.h"
#include "math.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <tuple>
#include <float_grad.h>
namespace cg = cooperative_groups;

namespace FORWARD
{

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
template <typename... JvpArgs>
__device__ auto computeColorFromSH(JvpArgs&&... jvp_args)
    // int idx, int deg, int max_coeffs, const glm::vec3* means, 
    // glm::vec3 campos, const float* shs, bool* clamped
{
    auto jvp_args_tuple = std::forward_as_tuple(std::forward<JvpArgs>(jvp_args)...);
    int idx = std::get<0>(jvp_args_tuple);
    int deg = std::get<1>(jvp_args_tuple);
    int max_coeffs = std::get<2>(jvp_args_tuple);
    auto means = std::get<3>(jvp_args_tuple);
    auto campos = std::get<4>(jvp_args_tuple);
    auto shs = std::get<5>(jvp_args_tuple);
    bool* clamped = std::get<6>(jvp_args_tuple);

    // The implementation is loosely based on code for 
    // "Differentiable Point-Based Radiance Fields for 
    // Efficient View Synthesis" by Zhang et al. (2022)
    auto pos = means[idx];
    auto dir = pos - campos;
    dir = dir / FLOATGRAD::length(dir);

    using ResultType = std::conditional_t<is_float_grad<decltype(dir)>::value
                                          || is_float_grad<decltype(shs)>::value,
                                          FloatGrad<glm::vec3>, glm::vec3>;
    auto sh = shs + idx * max_coeffs;
    ResultType result = SH_C0 * ResultType(sh[0], sh[1], sh[2]);

    if (deg > 0)
    {
        auto x = dir.x;
        auto y = dir.y;
        auto z = dir.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        if (deg > 1)
        {
            auto xx = x * x, yy = y * y, zz = z * z;
            auto xy = x * y, yz = y * z, xz = x * z;
            result = result +
                SH_C2[0] * xy * sh[4] +
                SH_C2[1] * yz * sh[5] +
                SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                SH_C2[3] * xz * sh[7] +
                SH_C2[4] * (xx - yy) * sh[8];

            if (deg > 2)
            {
                result = result +
                    SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                    SH_C3[1] * xy * z * sh[10] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                    SH_C3[5] * z * (xx - yy) * sh[14] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    result += 0.5f;

    // RGB colors are clamped to positive values. If values are
    // clamped, we need to keep track of this for the backward pass.
    clamped[3 * idx + 0] = (result.x < 0);
    clamped[3 * idx + 1] = (result.y < 0);
    clamped[3 * idx + 2] = (result.z < 0);
    return FLOATGRAD::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
template <typename... JvpArgs>
__device__ auto computeCov2D(JvpArgs&&... jvp_args)
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
    t.x = fminf(limx, fmaxf(-limx, txtz)) * t.z;
    t.y = fminf(limy, fmaxf(-limy, tytz)) * t.z;

    using Jtype = std::conditional_t<is_float_grad<decltype(t)>::value
                                     || is_float_grad<decltype(focal_x)>::value
                                     || is_float_grad<decltype(focal_y)>::value,
                                     FloatGrad<glm::mat3>, glm::mat3>;

    Jtype J = Jtype(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0.0f, 0.0f, 0.0f);

    using Wtype = std::conditional_t<is_float_grad<decltype(viewmatrix)>::value,
                                     FloatGrad<glm::mat3>, glm::mat3>;

    Wtype W = Wtype(
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    using Ttype = std::conditional_t<is_float_grad<Jtype>::value || is_float_grad<Wtype>::value,
                                     FloatGrad<glm::mat3>, glm::mat3>;

    Ttype T = W * J;

    using Vtype = std::conditional_t<is_float_grad<decltype(cov3D)>::value,
                                     FloatGrad<glm::mat3>, glm::mat3>;

    Vtype Vrk = Vtype(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    using Ctype = std::conditional_t<is_float_grad<Ttype>::value || is_float_grad<Vtype>::value,
                                     FloatGrad<glm::mat3>, glm::mat3>;

    Ctype cov = FLOATGRAD::transpose(T) * FLOATGRAD::transpose(Vrk) * T;

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;

    return make_float3(cov[0][0], cov[0][1], cov[1][1]);
}

template <typename T1, typename T2, typename T3, typename T4,
          typename = std::enable_if_t<is_float_grad<T1>::value
                                      || is_float_grad<T2>::value
                                      || is_float_grad<T3>::value
                                      || is_float_grad<T4>::value>>
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
    Rtype R = Rtype(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    using Mtype = std::conditional_t<is_float_grad<Stype>::value || is_float_grad<Rtype>::value, 
                                     FloatGrad<glm::mat3>, glm::mat3>;

    Mtype M = S * R;

    // Compute 3D world covariance matrix Sigma
    Mtype Sigma = FLOATGRAD::transpose(M) * M;

    // Covariance is symmetric, only store upper right
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C, typename TupleType>
__global__ void preprocessCUDAJvp(TupleType jvp_args_tuple)
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
    auto rgb = std::get<25>(jvp_args_tuple);
    auto conic_opacity = std::get<26>(jvp_args_tuple);
    const dim3 grid = std::get<27>(jvp_args_tuple);
    uint32_t* tiles_touched = std::get<28>(jvp_args_tuple);
    bool prefiltered = std::get<29>(jvp_args_tuple);

    static_assert(is_float_grad<decltype(cov3D_precomp)>::value 
                    == is_float_grad<decltype(cov3Ds)>::value, 
                  "Covariance matrices must be of the same type (either FloatGrad or not).");
    static_assert(is_float_grad<decltype(colors_precomp)>::value 
                    == is_float_grad<decltype(rgb)>::value, 
                  "Covariance matrices must be of the same type (either FloatGrad or not).");

    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Initialize radius and touched tiles to 0. If this isn't changed,
    // this Gaussian will not be processed further.
    radii[idx] = 0;
    tiles_touched[idx] = 0;

    // Perform near culling, quit if outside.
    using Ptype = std::conditional_t<is_float_grad<decltype(orig_points)>::value 
                                     || is_float_grad<decltype(viewmatrix)>::value
                                     || is_float_grad<decltype(projmatrix)>::value,
                                     FloatGrad<float3>, float3>;
    Ptype p_view(make_float3(0)); 
    if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
        return;

    // Transform point by projecting
    auto p_orig = make_float3(orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]);
    auto p_hom = transformPoint4x4(p_orig, projmatrix);
    auto p_w = 1.0f / (p_hom.w + 0.0000001f);
    auto p_proj = make_float3(p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w);

    // If 3D covariance matrix is precomputed, use it, otherwise compute
    // from scaling and rotation parameters. 
    using CovType = std::conditional_t<is_float_grad<decltype(cov3Ds)>::value,
                                       FloatGradArray<const float>, const float*>;
    CovType cov3D;
    if (get_data_ptr(cov3D_precomp) != nullptr)
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

    // Invert covariance (EWA algorithm)
    auto det = (cov.x * cov.z - cov.y * cov.y);
    if (det == 0.0f)
        return;
    auto det_inv = 1.f / det;
    auto conic = make_float3(cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv);

    // Compute extent in screen space (by finding eigenvalues of
    // 2D covariance matrix). Use extent to compute a bounding rectangle
    // of screen-space tiles that this Gaussian overlaps with. Quit if
    // rectangle covers 0 tiles. 
    auto mid = 0.5f * (cov.x + cov.z);
    auto lambda1 = mid + sqrtf(fmaxf(0.1f, mid * mid - det));
    auto lambda2 = mid - sqrtf(fmaxf(0.1f, mid * mid - det));
    auto my_radius = ceilf(3.f * sqrtf(fmaxf(lambda1, lambda2)));
    auto point_image = make_float2(ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H));

    uint2 rect_min, rect_max;
    getRect(get_data(point_image), get_data(my_radius), rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;

    // If colors have been precomputed, use them, otherwise convert
    // spherical harmonics coefficients to RGB color.
    if (get_data_ptr(colors_precomp) == nullptr)
    {
        auto result = computeColorFromSH(idx, D, M, cast<glm::vec3>(orig_points), 
                                         *cam_pos, shs, clamped);
        rgb[idx * C + 0] = result.x;
        rgb[idx * C + 1] = result.y;
        rgb[idx * C + 2] = result.z;
    }

    // Store some useful helper data for the next steps.
    depths[idx] = p_view.z;
    radii[idx] = get_data(my_radius);
    points_xy_image[idx] = point_image;
    // Inverse 2D covariance and opacity neatly pack into one float4
    conic_opacity[idx] = make_float4(conic.x, conic.y, conic.z, opacities[idx]);
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
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
    auto jvp_args_tuple = std::make_tuple(std::forward<JvpArgs>(jvp_args)...);
    int P = std::get<0>(jvp_args_tuple);
    
    preprocessCUDAJvp<CudaRasterizer::NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
            jvp_args_tuple
    );
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS, typename TupleType>
__global__ void __launch_bounds__(CudaRasterizer::BLOCK_X * CudaRasterizer::BLOCK_Y)
renderCUDAJvp(TupleType jvp_args_tuple)
    // const dim3 grid, dim3 block,
    // const uint2* ranges,
    // const uint32_t* point_list,
    // int W, int H,
    // const float2* points_xy_image,
    // const float4* conic_opacity,
    // float* final_T,
    // uint32_t* n_contrib,
    // const float* bg_color,
    // float* out_color,
    // const float* depth,
    // float* out_depth, 
    // float* out_opacity,
    // int* n_touched)
{

    // Unpack JVP arguments
    const uint2* ranges = std::get<2>(jvp_args_tuple);
    const uint32_t* point_list = std::get<3>(jvp_args_tuple);
    int W = std::get<4>(jvp_args_tuple);
    int H = std::get<5>(jvp_args_tuple);
    auto points_xy_image = std::get<6>(jvp_args_tuple);
    auto features = std::get<7>(jvp_args_tuple);
    auto conic_opacity = std::get<8>(jvp_args_tuple);
    auto final_T = std::get<9>(jvp_args_tuple);
    uint32_t* n_contrib = std::get<10>(jvp_args_tuple);
    auto bg_color = std::get<11>(jvp_args_tuple);
    auto out_color = std::get<12>(jvp_args_tuple);
    auto depth = std::get<13>(jvp_args_tuple);
    auto out_depth = std::get<14>(jvp_args_tuple);
    auto out_opacity = std::get<15>(jvp_args_tuple);
    int* n_touched = std::get<16>(jvp_args_tuple);

    constexpr int BLOCK_X = CudaRasterizer::BLOCK_X;
    constexpr int BLOCK_Y = CudaRasterizer::BLOCK_Y;
    constexpr int BLOCK_SIZE = CudaRasterizer::BLOCK_SIZE;


    // Identify current tile and associated min/max pixel range.
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    // uint32_t horizontal_blocks = gridDim.x; # TODO Maybe it's different?
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = { (float)pix.x, (float)pix.y };

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W&& pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    // Allocate storage for batches of collectively fetched data.
    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy_data[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity_data[BLOCK_SIZE];
    __shared__ float collected_depth_data[BLOCK_SIZE];
    __shared__ float2 collected_xy_grad[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity_grad[BLOCK_SIZE];
    __shared__ float collected_depth_grad[BLOCK_SIZE];

    FloatGradArray<float2> collected_xy(collected_xy_data, collected_xy_grad);
    FloatGradArray<float4> collected_conic_opacity(collected_conic_opacity_data, collected_conic_opacity_grad);
    FloatGradArray<float> collected_depth(collected_depth_data, collected_depth_grad);

    // Initialize helper variables
    FloatGrad<float> T = 1.0f;
    uint32_t contributor = 0;
    uint32_t last_contributor = 0;
    float C_data[CHANNELS] = { 0 }, C_grad[CHANNELS] = { 0 };
    FloatGradArray<float> C(C_data, C_grad);
    FloatGrad<float> D = 0.0f;

    // Iterate over batches until all done or range is complete
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // End if entire block votes that it is done rasterizing
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        // Collectively fetch per-Gaussian data from global to shared
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            collected_depth[block.thread_rank()] = depth[coll_id];
        }
        block.sync();

        // Iterate over current batch
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {

            // Keep track of current position in range
            contributor++;

            // Resample using conic matrix (cf. "Surface 
            // Splatting" by Zwicker et al., 2001)
            FloatGrad<float2> xy = collected_xy[j];
            FloatGrad<float2> d = make_float2(xy.x - pixf.x, xy.y - pixf.y);
            FloatGrad<float4> con_o = collected_conic_opacity[j];
            FloatGrad<float> power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f)
                continue;

            // Eq. (2) from 3D Gaussian splatting paper.
            // Obtain alpha by multiplying with Gaussian opacity
            // and its exponential falloff from mean.
            // Avoid numerical instabilities (see paper appendix). 
            FloatGrad<float> alpha = fminf(0.99f, con_o.w * expf(power));

            if (alpha < 1.0f / 255.0f) {
                continue;
            }
            FloatGrad<float> test_T = T * (1 - alpha);

            if (test_T < 0.0001f)
            {
                done = true;
                continue;
            }
            // Eq. (3) from 3D Gaussian splatting paper.
            for (int ch = 0; ch < CHANNELS; ch++) {
                C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
            }
            D += collected_depth[j] * alpha * T;
            // Keep track of how many pixels touched this Gaussian.
            if (test_T > 0.5f) {
                atomicAdd(&(n_touched[collected_id[j]]), 1);
            }
            T = test_T;

            // Keep track of last range entry to update this
            // pixel.
            last_contributor = contributor;
        }
    }

    // TODO: There is something weird with assigment operators again so that out_opacity is not assigned properly

    // All threads that treat valid pixel write out their final
    // rendering data to the frame and auxiliary buffers.
    if (inside)
    {
        final_T[pix_id] = T;
        n_contrib[pix_id] = last_contributor;
        for (int ch = 0; ch < CHANNELS; ch++) {
            out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
        }
        out_depth[pix_id] = D;
        out_opacity[pix_id] = 1 - T;
    }
}

template <typename... JvpArgs>
void renderJvp(JvpArgs&&... jvp_args)
    // const dim3 grid, dim3 block,
    // const uint2* ranges,
    // const uint32_t* point_list,
    // int W, int H,
    // const float2* means2D,
    // const float* colors,
    // const float4* conic_opacity,
    // float* final_T,
    // uint32_t* n_contrib,
    // const float* bg_color,
    // float* out_color,
    // const float* depth,
    // float* out_depth, 
    // float* out_opacity,
    // int* n_touched)
{
    constexpr int NUM_CHANNELS = CudaRasterizer::NUM_CHANNELS;

    auto jvp_args_tuple = std::make_tuple(std::forward<JvpArgs>(jvp_args)...);
    auto grid = std::get<0>(jvp_args_tuple);
    auto block = std::get<1>(jvp_args_tuple);

    renderCUDAJvp<NUM_CHANNELS> << <grid, block >> > (jvp_args_tuple);
}

} // namespace FORWARD

#endif // FORWARD_IMPL_H
