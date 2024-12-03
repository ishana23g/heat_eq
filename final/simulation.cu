// simulation.cpp

#include "simulation.h"
#include "constants.h"
#include "utility.h"

void init_simulation(float **d_u0, float **d_u1, size_t *size, SimulationMode mode)
{
    if (mode == MODE_1D)
    {
        *size = WIDTH * sizeof(float);
    }
    else if (mode == MODE_2D)
    {
        *size = WIDTH * HEIGHT * sizeof(float);
    }
    else if (mode == MODE_3D)
    {
        *size = WIDTH * HEIGHT * DEPTH * sizeof(float);
    }
    gpuErrchk(cudaMalloc((void **)d_u0, *size));
    gpuErrchk(cudaMalloc((void **)d_u1, *size));

    gpuErrchk(cudaMemset(*d_u0, 0, *size));
    gpuErrchk(cudaMemset(*d_u1, 0, *size));
}

void reset_simulation(float *d_u0, float *d_u1, size_t size)
{
    cudaMemset(d_u0, 0, size);
    cudaMemset(d_u1, 0, size);
}

void simulate(float *d_u0, float *d_u1, size_t size, uchar4 *d_output, SimulationMode simulation_mode, BoundaryCondition boundary_condition)
{
    if (simulation_mode == MODE_1D)
    {
        dim3 blockSize(256);
        dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x);
        int sharedMemBytes = (blockSize.x + 2) * sizeof(float);

        heat_kernel_1d<<<gridSize, blockSize, sharedMemBytes>>>(d_u0, d_u1, WIDTH, TIME_STEP, DX * DX, DIFFUSIVITY, boundary_condition);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        compute_output_kernel_1d<<<gridSize, blockSize>>>(d_u1, d_output, WIDTH, simulation_mode);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
    else if (simulation_mode == MODE_2D)
    {
        dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 gridSize((WIDTH + blockSize.x -1) / blockSize.x, (HEIGHT + blockSize.y -1) / blockSize.y);
        int sharedMemBytes = (blockSize.x + 2) * (blockSize.y +2) * sizeof(float);

        heat_kernel_2d<<<gridSize, blockSize, sharedMemBytes>>>(d_u0, d_u1, WIDTH, HEIGHT, TIME_STEP, DX * DX, DY * DY, DIFFUSIVITY, boundary_condition);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        compute_output_kernel_2d<<<gridSize, blockSize>>>(d_u1, d_output, WIDTH, HEIGHT, simulation_mode);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
    else if (simulation_mode == MODE_3D)
    {
        dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
        dim3 gridSize((WIDTH + blockSize.x -1) / blockSize.x, (HEIGHT + blockSize.y -1) / blockSize.y, DEPTH);

        heat_kernel_3d<<<gridSize, blockSize>>>(d_u0, d_u1, WIDTH, HEIGHT, DEPTH, TIME_STEP, DX * DX, DY * DY, DZ * DZ, DIFFUSIVITY, boundary_condition);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        dim3 blockSize(BLOCK_SIZE_X/2, BLOCK_SIZE_Y/2, BLOCK_SIZE_Z/2);
        dim3 gridSize((WIDTH + blockSize.x -1) / blockSize.x, 
            (HEIGHT + blockSize.y -1) / blockSize.y, 
            (DEPTH + blockSize.z -1) / blockSize.z);

        compute_output_kernel_3d<<<gridSize, blockSize>>>(d_u1, d_output, WIDTH, HEIGHT, DEPTH, simulation_mode);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    float *temp = d_u0;
    d_u0 = d_u1;
    d_u1 = temp;
}

void add_heat_launcher(float *d_u0, SimulationMode simulation_mode, double xpos, double ypos, int window_width, int window_height)
{
    if (simulation_mode == MODE_1D)
    {
        int x = (int)(xpos * WIDTH / window_width);

        dim3 blockSize(256);
        dim3 gridSize((2 * HEAT_RADIUS + blockSize.x - 1) / blockSize.x);

        add_heat_kernel_1d<<<gridSize, blockSize>>>(d_u0, WIDTH, x);
    }
    else if (simulation_mode == MODE_2D)
    {
        int x = (int)(xpos * WIDTH / window_width);
        int y = (int)((window_height - ypos) * HEIGHT / window_height);

        dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 gridSize((2 * HEAT_RADIUS + blockSize.x - 1) / blockSize.x,
                        (2 * HEAT_RADIUS + blockSize.y - 1) / blockSize.y);

        add_heat_kernel_2d<<<gridSize, blockSize>>>(d_u0, WIDTH, HEIGHT, x, y);
    }
    else if (simulation_mode == MODE_3D)
    {
        int x = (int)(xpos * WIDTH / window_width);
        int y = (int)((window_height - ypos) * HEIGHT / window_height);
        int z = rand() % DEPTH;

        dim3 blockSize(BLOCK_SIZE_X / 2, BLOCK_SIZE_Y / 2, BLOCK_SIZE_Z / 2);
        dim3 gridSize((2 * HEAT_RADIUS + blockSize.x - 1) / blockSize.x,
                        (2 * HEAT_RADIUS + blockSize.y - 1) / blockSize.y,
                        (2 * HEAT_RADIUS + blockSize.z - 1) / blockSize.z);

        add_heat_kernel_3d<<<gridSize, blockSize>>>(d_u0, WIDTH, HEIGHT, DEPTH, x, y, z);
    }
    cudaDeviceSynchronize();
}