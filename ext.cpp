/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "rasterize_points.h"

class GPUTimer {
    cudaEvent_t start_evt, stop_evt;
public:
    GPUTimer() {
        cudaEventCreate(&start_evt);
        cudaEventCreate(&stop_evt);
    }
    void start() {
        cudaEventRecord(start_evt);
    }
    float stop_clock_get_elapsed() {
        float time_millis;
        cudaEventRecord(stop_evt);
        cudaEventSynchronize(stop_evt);
        cudaEventElapsedTime(&time_millis, start_evt, stop_evt);
        return time_millis;
    }
    ~GPUTimer() {
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
    }
};



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<GPUTimer>(m, "GPUTimer")
        .def(py::init<>())
        .def("start", &GPUTimer::start)
        .def("stop_clock_get_elapsed", &GPUTimer::stop_clock_get_elapsed);
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("rasterize_gaussians_backward_sketch_jacobian", &RasterizeGaussiansBackwardSketchJacobianCUDA);
  m.def("mark_visible", &markVisible);
}
