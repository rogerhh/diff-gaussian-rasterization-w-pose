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

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        if num_backward_gaussians > 0:
            num_backward_gaussians = min(num_backward_gaussians, radii.shape[0])
        ctx.num_backward_gaussians = num_backward_gaussians
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)

        # print("num_backward_gaussians: ", num_backward_gaussians)

        if num_backward_gaussians > 0:
            # Add a small epsilon to avoid zero probabilities
            # ctx.index_dist = (radii.float() / torch.sum(radii.float()) + 1e-8)
            ctx.index_dist = n_touched.float() / (n_touched.float() + 1e-8)
            ctx.index_dist = (ctx.index_dist / ctx.index_dist.sum() + 1e-8) * num_backward_gaussians

            ctx.selected_indices = torch.multinomial(ctx.index_dist, num_backward_gaussians, replacement=False)
            ctx.selected_bools = torch.zeros_like(radii, dtype=torch.bool)
            ctx.selected_bools[ctx.selected_indices] = True

        # Initialize some context variables that may be modified from outside
        ctx.tracking = False
        ctx.select_pixels = False
        ctx.selected_pixel_indices = None

        return color, radii, depth, opacity, n_touched

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_radii, grad_out_depth, grad_out_opacity, grad_n_touched):

        backward_start = time.time()

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        select_pixels = ctx.select_pixels
        selected_pixel_indices = ctx.selected_pixel_indices if select_pixels else torch.zeros(0, dtype=torch.int32)
        num_backward_gaussians = ctx.num_backward_gaussians
        select_gaussians = num_backward_gaussians > 0
        selected_indices = ctx.selected_indices if select_gaussians else torch.zeros(0, dtype=torch.int32)
        selected_bools = ctx.selected_bools if select_gaussians else torch.zeros(0, dtype=torch.bool)
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

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
                selected_pixel_indices.to(torch.int32),
                select_gaussians,
                selected_indices.to(torch.int32),
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
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_tau = _C.rasterize_gaussians_backward(*args)

            # if select_pixels:
            #     print("selected_pixel_indices: ", selected_pixel_indices)
            #     print("colors_precomp.shape: ", colors_precomp.shape)
            #     print("grad_out_color.shape: ", grad_out_color.shape)
            #     print("grad_out_color: ", grad_out_color)
            #     sample_grad_means3D = grad_means3D.clone()
            #     sample_grad_tau = grad_tau.clone()
            #     print("select pixels grad_means3D: ", grad_means3D)

            #     args = (raster_settings.bg,
            #             means3D,
            #             radii,
            #             colors_precomp,
            #             scales,
            #             rotations,
            #             raster_settings.scale_modifier,
            #             cov3Ds_precomp,
            #             raster_settings.viewmatrix,
            #             raster_settings.projmatrix,
            #             raster_settings.projmatrix_raw,
            #             raster_settings.tanfovx,
            #             raster_settings.tanfovy,
            #             grad_out_color,
            #             grad_out_depth,
            #             sh,
            #             raster_settings.sh_degree,
            #             raster_settings.campos,
            #             geomBuffer,
            #             num_rendered,
            #             binningBuffer,
            #             imgBuffer,
            #             False,
            #             selected_pixel_indices.to(torch.int32),
            #             select_gaussians,
            #             selected_indices.to(torch.int32),
            #             selected_bools,
            #             raster_settings.debug)

            #     grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_tau = _C.rasterize_gaussians_backward(*args)

            #     print("correct grad_means3D: ", grad_tau)
            #     print(f"diff max: {(grad_tau - sample_grad_tau).abs().max()}")

            #     import code; code.interact(local=locals())

        if select_gaussians:
            selected_indices = ctx.selected_indices
            index_dist = ctx.index_dist

            # DEBUG
            grad_tau_gt = torch.sum(grad_tau.view(-1, 6), dim=0)
            # DEBUG END

            # grad_tau[selected_indices] /= index_dist[selected_indices].view(-1, 1)
            # grad_tau = torch.sum(grad_tau.view(-1, 6)[selected_indices], dim=0)
            # grad_tau /= num_backward_gaussians

            norms = grad_tau.norm(dim=1) + 1e-8
            norms = norms / norms.sum()
            print("norms.shape: ", norms.shape)
            selected_indices = torch.multinomial(norms, num_backward_gaussians, replacement=True)
            grad_tau[selected_indices] /= norms[selected_indices].view(-1, 1)
            grad_tau = torch.sum(grad_tau.view(-1, 6)[selected_indices], dim=0)
            grad_tau /= num_backward_gaussians


            # DEBUG
            diff = grad_tau_gt - grad_tau
            eps = diff.norm() / grad_tau_gt.norm()
            print("grad_tau_gt: ", grad_tau_gt)
            print("grad_tau: ", grad_tau)
            print("eps: ", eps.item())

            # DEBUG END

        else:
            grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)

        grad_rho = grad_tau[:3].view(1, -1)
        grad_theta = grad_tau[3:].view(1, -1)

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
        )

        backward_end = time.time()
        backward_time_ms = (backward_end - backward_start) * 1000

        # # DEBUG
        # if select_pixels:
        #     print(f"grad_tau.nonzero().shape: {grad_tau.nonzero().shape}")


        # if ctx.tracking:
        #     print(f"backward time ms: {(backward_end - backward_start) * 1000}")

        # # DEBUG END

        ctx.stats = {"backward_time_ms": backward_time_ms}

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

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, theta=None, rho=None, num_backward_gaussians=None):
        
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
            num_backward_gaussians=num_backward_gaussians
        )

