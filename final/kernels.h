// kernels.h

#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>
#include "constants.h"

enum BoundaryCondition
{
    DIRICHLET,
    NEUMANN
};

enum SimulationMode
{
    MODE_1D,
    MODE_2D,
    MODE_3D
};

__global__ void heat_kernel_1d(float *u0, float *u1, int width, float dt, float dx2, float a, BoundaryCondition boundary_condition);
__global__ void heat_kernel_2d(float *u0, float *u1, int width, int height, float dt, float dx2, float dy2, float a, BoundaryCondition boundary_condition);
__global__ void heat_kernel_3d(float *u0, float *u1, int width, int height, int depth, float dt, float dx2, float dy2, float dz2, float a, BoundaryCondition boundary_condition);

__global__ void compute_output_kernel_1d(float *u, uchar4 *output, int width, SimulationMode mode);
__global__ void compute_output_kernel_2d(float *u, uchar4 *output, int width, int height, SimulationMode mode);
__global__ void compute_output_kernel_3d(float *u, uchar4 *output, int width, int height, int depth, SimulationMode mode);

__global__ void add_heat_kernel_1d(float *u, int width, int x);
__global__ void add_heat_kernel_2d(float *u, int width, int height, int cx, int cy);
__global__ void add_heat_kernel_3d(float *u, int width, int height, int depth, int cx, int cy, int cz);

#endif // KERNELS_H