#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C
import time

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    theta,
    rho,
    raster_settings,
    num_backward_gaussians=-1,
    sketch_mode=0,
    sketch_dim=0,
    stack_dim=0,
    sketch_dtau=None,
    sketch_indices=None,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
        num_backward_gaussians,
        sketch_mode,
        sketch_dim,
        stack_dim,
        sketch_dtau,
        sketch_indices,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
        num_backward_gaussians,
        sketch_mode,
        sketch_dim,
        stack_dim,
        sketch_dtau,
        sketch_indices,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depth, opacity, n_touched = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depth, opacity, n_touched = _C.rasterize_gaussians(*args)

        ctx.sketch_mode = sketch_mode
        ctx.sketch_dim = sketch_dim
        ctx.stack_dim = stack_dim
        ctx.repeat_iter = 0

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        if num_backward_gaussians > 0:
            num_backward_gaussians = min(num_backward_gaussians, radii.shape[0])
        ctx.num_backward_gaussians = num_backward_gaussians
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, sketch_dtau, sketch_indices, )

        """
        # THIS SECTION CURRENTLY DISABLED

        if num_backward_gaussians > 0:
            # Add a small epsilon to avoid zero probabilities
            # ctx.index_dist = (radii.float() / torch.sum(radii.float()) + 1e-8)
            ctx.index_dist = n_touched.float() / (n_touched.float() + 1e-8)
            ctx.index_dist = (ctx.index_dist / ctx.index_dist.sum() + 1e-8) * num_backward_gaussians

            ctx.selected_indices = torch.multinomial(ctx.index_dist, num_backward_gaussians, replacement=False)
            ctx.selected_bools = torch.zeros_like(radii, dtype=torch.bool)
            ctx.selected_bools[ctx.selected_indices] = True
        """

        return color, radii, depth, opacity, n_touched

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_radii, grad_out_depth, grad_out_opacity, grad_n_touched):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, sketch_dtau, sketch_indices, = ctx.saved_tensors

        sketch_mode = ctx.sketch_mode
        sketch_dim = ctx.sketch_dim
        stack_dim = ctx.stack_dim
        repeat_iter = ctx.repeat_iter

        select_pixels = False
        selected_pixel_indices = torch.zeros(0, dtype=torch.int32)
        select_gaussians = False
        selected_indices = torch.zeros(0, dtype=torch.int32)
        selected_bools = torch.zeros(0, dtype=torch.bool)

        if not raster_settings.debug and sketch_mode != 0:
            pass
        else:
            # Restructure args as C++ method expects them
            args = (raster_settings.bg,
                    means3D,
                    radii,
                    colors_precomp,
                    scales,
                    rotations,
                    raster_settings.scale_modifier,
                    cov3Ds_precomp,
                    raster_settings.viewmatrix,
                    raster_settings.projmatrix,
                    raster_settings.projmatrix_raw,
                    raster_settings.tanfovx,
                    raster_settings.tanfovy,
                    grad_out_color,
                    grad_out_depth,
                    sh,
                    raster_settings.sh_degree,
                    raster_settings.campos,
                    geomBuffer,
                    num_rendered,
                    binningBuffer,
                    imgBuffer,
                    select_pixels,
                    selected_pixel_indices,
                    select_gaussians,
                    selected_indices,
                    selected_bools,
                    raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_tau = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            if sketch_mode == 0:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_tau = _C.rasterize_gaussians_backward(*args)
            else:
                # sketch_grad_tau = torch.zeros((sketch_dtau.shape), device=sketch_dtau.device)
                sketch_grad_tau = torch.empty((sketch_dtau.shape), device=sketch_dtau.device)

                for j in range(stack_dim):
                    args = (raster_settings.bg,
                            means3D,
                            radii,
                            colors_precomp,
                            scales,
                            rotations,
                            raster_settings.scale_modifier,
                            cov3Ds_precomp,
                            raster_settings.viewmatrix,
                            raster_settings.projmatrix,
                            raster_settings.projmatrix_raw,
                            raster_settings.tanfovx,
                            raster_settings.tanfovy,
                            grad_out_color,
                            grad_out_depth,
                            sh,
                            raster_settings.sh_degree,
                            raster_settings.campos,
                            geomBuffer,
                            num_rendered,
                            binningBuffer,
                            imgBuffer,
                            select_pixels,
                            selected_pixel_indices,
                            select_gaussians,
                            selected_indices,
                            selected_bools,
                            sketch_mode,
                            sketch_dim,
                            sketch_indices[repeat_iter, j],
                            raster_settings.debug)

                    assert(sketch_dim <= 64)

                    sketch_grad_means2D_j, sketch_grad_colors_precomp_j, sketch_grad_opacities_j, sketch_grad_means3D_j, sketch_grad_cov3Ds_precomp_j, sketch_grad_sh_j, sketch_grad_scales_j, sketch_grad_rotations_j, sketch_grad_tau_j = _C.rasterize_gaussians_backward_sketch_jacobian(*args)

                    # torch.cuda.synchronize()

                    # sum_start = time.time()
                    # try:
                    #     chunk_size = 65536 # 2^16
                    #     sum_dim = sketch_grad_tau_j.shape[0]
                    #     chunk_start = 0
                    #     while chunk_start < sum_dim:
                    #         if sum_dim - chunk_start >= 2 * chunk_size:
                    #             chunk_end = chunk_start + chunk_size
                    #         else:
                    #             chunk_end = sum_dim
                    #         sketch_grad_tau[j] += sketch_grad_tau_j[chunk_start:chunk_end].sum(dim=0)
                    #         chunk_start = chunk_end

                    # except:
                    #     print("Error in sketch_grad_tau")
                    #     import code; code.interact(local=locals())
                    # torch.cuda.synchronize()
                    # sum_end = time.time()

                    try:
                        sketch_grad_tau[j] = sketch_grad_tau_j.sum(dim=0)
                    except:
                        print("Error in sketch_grad_tau")
                        import code; code.interact(local=locals())


                    """
                    for i in range(sketch_dim):
                        print(f"i, j = {i}, {j}")
                        grad_out_color_i = torch.zeros_like(grad_out_color, device=grad_out_color.device)
                        grad_out_depth_i = torch.zeros_like(grad_out_depth, device=grad_out_depth.device)
                        indices_i = sketch_indices[repeat_iter, j] == i
                        grad_out_color_i[:, indices_i] = grad_out_color[:, indices_i]
                        grad_out_depth_i[:, indices_i] = grad_out_depth[:, indices_i]
                        args = (raster_settings.bg,
                                means3D,
                                radii,
                                colors_precomp,
                                scales,
                                rotations,
                                raster_settings.scale_modifier,
                                cov3Ds_precomp,
                                raster_settings.viewmatrix,
                                raster_settings.projmatrix,
                                raster_settings.projmatrix_raw,
                                raster_settings.tanfovx,
                                raster_settings.tanfovy,
                                grad_out_color_i,
                                grad_out_depth_i,
                                sh,
                                raster_settings.sh_degree,
                                raster_settings.campos,
                                geomBuffer,
                                num_rendered,
                                binningBuffer,
                                imgBuffer,
                                select_pixels,
                                selected_pixel_indices.to(torch.int32),
                                select_gaussians,
                                selected_indices.to(torch.int32),
                                selected_bools,
                                raster_settings.debug)

                        correct_grad_means2D, correct_grad_colors_precomp, correct_grad_opacities, correct_grad_means3D, correct_grad_cov3Ds_precomp, correct_grad_sh, correct_grad_scales, correct_grad_rotations, correct_grad_tau = _C.rasterize_gaussians_backward(*args)

                        assert(torch.allclose(sketch_grad_means2D_j[:, i, ...], correct_grad_means2D, atol=1e-5, rtol=1e-5))
                        assert(torch.allclose(sketch_grad_colors_precomp_j[:, i, ...], correct_grad_colors_precomp, atol=1e-5, rtol=1e-5)), f"i = {i}, {sketch_grad_colors_precomp_j[i]} != {correct_grad_colors_precomp}"
                        assert(torch.allclose(sketch_grad_opacities_j[:, i, ...], correct_grad_opacities, atol=1e-5, rtol=1e-5))
                        assert torch.allclose(sketch_grad_means3D_j[:, i, ...], correct_grad_means3D, atol=1e-5, rtol=1e-5), f"i = {i}, {sketch_grad_means3D_j[:, i, ...]} != {correct_grad_means3D}"
                        assert(torch.allclose(sketch_grad_cov3Ds_precomp_j[:, i, ...], correct_grad_cov3Ds_precomp, atol=1e-3, rtol=1e-3))
                        assert(torch.allclose(sketch_grad_sh_j[:, i, ...], correct_grad_sh, atol=1e-5, rtol=1e-5))
                        assert torch.allclose(sketch_grad_scales_j[:, i, ...], correct_grad_scales, atol=1e-5, rtol=1e-5), f"{sketch_grad_scales_j[i]} != {correct_grad_scales}"
                        assert(torch.allclose(sketch_grad_rotations_j[:, i, ...], correct_grad_rotations, atol=1e-5, rtol=1e-5))
                        assert(torch.allclose(sketch_grad_tau_j[:, i, ...], correct_grad_tau, atol=1e-5, rtol=1e-5))

                        print("Sketch Jacobian test passed")
                        """

                # grad_means2D = torch.sum(sketch_grad_means2D, dim=1)
                # grad_colors_precomp = torch.sum(sketch_grad_colors_precomp, dim=1)
                # grad_opacities = torch.sum(sketch_grad_opacities, dim=1)
                # grad_means3D = torch.sum(sketch_grad_means3D, dim=1)
                # grad_cov3Ds_precomp = torch.sum(sketch_grad_cov3Ds_precomp, dim=1)
                # grad_sh = torch.sum(sketch_grad_sh, dim=1)
                # grad_scales = torch.sum(sketch_grad_scales, dim=1)
                # grad_rotations = torch.sum(sketch_grad_rotations, dim=1)
                # grad_tau = torch.sum(sketch_grad_tau, dim=1)

                grad_means2D = None
                grad_colors_precomp = None
                grad_opacities = None
                grad_means3D = None
                grad_cov3Ds_precomp = None
                grad_sh = None
                grad_scales = None
                grad_rotations = None
                grad_tau = None
                

        if grad_tau is not None:
            if select_gaussians:

                norms = grad_tau.norm(dim=1) + 1e-8
                norms = norms / norms.sum()
                selected_indices = torch.multinomial(norms, num_backward_gaussians, replacement=True)
                grad_tau[selected_indices] /= norms[selected_indices].view(-1, 1)
                grad_tau = torch.sum(grad_tau.view(-1, 6)[selected_indices], dim=0)
                grad_tau /= num_backward_gaussians

            else:
                grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)

            grad_rho = grad_tau[:3].view(1, -1)
            grad_theta = grad_tau[3:].view(1, -1)
        else:
            grad_rho = None
            grad_theta = None

        grad_sketch_mode = None
        grad_sketch_dim = None
        grad_stack_dim = None
        grad_sketch_dtau = sketch_grad_tau if sketch_mode != 0 else None
        grad_sketch_indices = None

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_theta,
            grad_rho,
            None,
            None,
            grad_sketch_mode,
            grad_sketch_dim,
            grad_stack_dim,
            grad_sketch_dtau,
            grad_sketch_indices,
        )

        ctx.repeat_iter += 1

        # # DEBUG
        # if select_pixels:
        #     print(f"grad_tau.nonzero().shape: {grad_tau.nonzero().shape}")


        # if ctx.tracking:
        #     print(f"rasterize_gaussians_backward_time_ms: {rasterize_gaussians_backward_time_ms}")
        #     print(f"rasterize_gaussians_cuda_backward_time_ms: {rasterize_gaussians_cuda_backward_time_ms}")
        #     print(f"rasterize_gaussians_C_backward_time_ms: {rasterize_gaussians_C_backward_time_ms}")
        #     # print(f"sum_tau_time_ms: {(sum_tau_end - rasterize_gaussians_C_backward_end) * 1000}")
        #     # print(f"prep arg time: {(rasterize_gaussians_C_backward_start - rasterize_gaussians_backward_start) * 1000}")


        # # DEBUG END

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    projmatrix_raw : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, theta=None, rho=None, num_backward_gaussians=None, sketch_mode=0, sketch_dim=0, stack_dim=0, sketch_dtau=None, sketch_indices=None,):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if theta is None:
            theta = torch.Tensor([])
        if rho is None:
            rho = torch.Tensor([])
        

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            theta,
            rho,
            raster_settings, 
            num_backward_gaussians=num_backward_gaussians,
            sketch_mode=sketch_mode,
            sketch_dim=sketch_dim,
            stack_dim=stack_dim,
            sketch_dtau=sketch_dtau,
            sketch_indices=sketch_indices,
        )

