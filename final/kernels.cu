// kernels.cu

#include "kernels.h"
#include "constants.h"

// Indexing macro
__device__ int IDX_2D(int x, int y, int width) {
    return y * width + x;
}

__device__ int IDX_3D(int x, int y, int z, int width, int height) {
    return z * width * height + y * width + x;
}

__device__ float clamp(float x) {
    return x < HEAT_MIN_CLAMP ? HEAT_MIN_CLAMP : (x > HEAT_MAX_CLAMP ? HEAT_MAX_CLAMP : x);
}

__device__ void gradient_scaling(float heat_value, uchar4* out_color, SimulationMode mode)
{
    float t = clamp(heat_value / HEAT_SOURCE);
    unsigned char a = static_cast<unsigned char>(255.0f);

    if (mode == MODE_3D) {
        a = static_cast<unsigned char>(255.0f / DEPTH * 4.0f);
    }

    unsigned char r = static_cast<unsigned char>(t * 255.0f);
    unsigned char g = static_cast<unsigned char>(0);
    unsigned char b = static_cast<unsigned char>(255.0f - t * 255.0f);

    *out_color = make_uchar4(r, g, b, a);
}

__global__ void heat_kernel_1d(float *u0, float *u1, int width, float dt, float dx2, float a, BoundaryCondition boundary_condition)
{
    extern __shared__ float s_u[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int s_x = threadIdx.x + 1;

    if (x < width) {
        s_u[s_x] = u0[x];

        if (threadIdx.x == 0 && x > 0)
            s_u[0] = u0[x - 1];
        if (threadIdx.x == blockDim.x - 1 && x < width - 1)
            s_u[blockDim.x + 1] = u0[x + 1];

        __syncthreads();

        if (x > 0 && x < width -1) {
            float u_center = s_u[s_x];
            float u_left = s_u[s_x - 1];
            float u_right = s_u[s_x + 1];

            float laplacian = (u_left - 2 * u_center + u_right) / dx2;

            u1[x] = u_center + a * dt * laplacian;
        }
        else if (x == 0 || x == width -1) {
            switch (boundary_condition)
            {
            case DIRICHLET:
                u1[x] = 0.0f;
                break;
            case NEUMANN:
                if (x == 0)
                    u1[x] = u1[x + 1] + HEAT_SOURCE * dx2;
                else if (x == width - 1)
                    u1[x] = u1[x - 1] + HEAT_SOURCE * dx2;
                break;
            }
        }
    }
}

__global__ void compute_output_kernel_1d(float *u, uchar4 *output, int width, SimulationMode mode)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < width) {
        gradient_scaling(u[x], &output[x], mode);
    }
}

__global__ void heat_kernel_2d(float *u0, float *u1, int width, int height, float dt, float dx2, float dy2, float a, BoundaryCondition boundary_condition)
{
    extern __shared__ float s_u[];

    int shared_width = blockDim.x + 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int s_x = threadIdx.x + 1;
    int s_y = threadIdx.y + 1;

    if (x < width && y < height) {
        s_u[s_y * shared_width + s_x] = u0[y * width + x];

        if (threadIdx.x == 0 && x > 0)
            s_u[s_y * shared_width + 0] = u0[y * width + x -1];
        if (threadIdx.x == blockDim.x -1 && x < width -1)
            s_u[s_y * shared_width + s_x +1] = u0[y * width + x +1];
        if (threadIdx.y == 0 && y > 0)
            s_u[ (s_y -1) * shared_width + s_x ] = u0[(y -1) * width + x];
        if (threadIdx.y == blockDim.y -1 && y < height -1)
            s_u[ (s_y +1) * shared_width + s_x ] = u0[(y +1) * width + x];

        __syncthreads();

        if (x > 0 && x < width -1 && y > 0 && y < height -1) {
            float u_center = s_u[s_y * shared_width + s_x];
            float u_left = s_u[s_y * shared_width + s_x -1];
            float u_right = s_u[s_y * shared_width + s_x +1];
            float u_up = s_u[ (s_y -1) * shared_width + s_x ];
            float u_down = s_u[ (s_y +1) * shared_width + s_x ];

            float laplacian = (u_left - 2 * u_center + u_right ) / dx2 +
                              (u_up - 2 * u_center + u_down ) / dy2;

            u1[y * width + x] = u_center + a * dt * laplacian;
        }
        else if (x == 0 || x == width -1 || y == 0 || y == height -1) {
            int idx = y * width + x;
            switch (boundary_condition)
            {
            case DIRICHLET:
                u1[idx] = 0.0f;
                break;
            case NEUMANN:
                break;
            }
        }
    }
}

__global__ void compute_output_kernel_2d(float *u, uchar4 *output, int width, int height, SimulationMode mode)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = IDX_2D(x, y, width);
        gradient_scaling(u[idx], &output[idx], mode);
    }
}

__global__ void heat_kernel_3d(float *u0, float *u1, int width, int height, int depth, float dt, float dx2, float dy2, float dz2, float a, BoundaryCondition boundary_condition)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if (x >= 1 && x < width -1 && y >=1 && y < height -1 && z >=1 && z < depth -1) {
        int idx = IDX_3D(x, y, z, width, height);

        float u_center = u0[idx];
        float u_left = u0[IDX_3D(x -1, y, z, width, height)];
        float u_right = u0[IDX_3D(x +1, y, z, width, height)];
        float u_up = u0[IDX_3D(x, y -1, z, width, height)];
        float u_down = u0[IDX_3D(x, y +1, z, width, height)];
        float u_front = u0[IDX_3D(x, y, z -1, width, height)];
        float u_back = u0[IDX_3D(x, y, z +1, width, height)];

        float laplacian = (u_left - 2 * u_center + u_right) / dx2 +
                          (u_up - 2 * u_center + u_down) / dy2 +
                          (u_front - 2 * u_center + u_back) / dz2;

        u1[idx] = u_center + a * dt * laplacian;
    }
    else if (x == 0 || x == width -1 || y == 0 || y == height -1 || z == 0 || z == depth -1) {
        int idx = IDX_3D(x, y, z, width, height);
        switch (boundary_condition)
        {
        case DIRICHLET:
            u1[idx] = 0.0f;
            break;
        case NEUMANN:   

            


            break;
        }
    }
}

__global__ void compute_output_kernel_3d(float *u, uchar4 *output, int width, int height, int depth, SimulationMode mode)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        int idx = IDX_3D(x, y, z, width, height);
        gradient_scaling(u[idx], &output[idx], mode);
    }
}

__global__ void add_heat_kernel_1d(float *u, int width, int x)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x - HEAT_RADIUS;
    int idx = x + tx;

    if (idx >= 0 && idx < width)
    {
        if (abs(tx) <= HEAT_RADIUS)
        {
            u[idx] += HEAT_SOURCE;
        }
    }
}

__global__ void add_heat_kernel_2d(float *u, int width, int height, int cx, int cy)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x - HEAT_RADIUS;
    int ty = blockIdx.y * blockDim.y + threadIdx.y - HEAT_RADIUS;

    int x = cx + tx;
    int y = cy + ty;

    if (x >= 0 && x < width && y >= 0 && y < height)
    {
        if (tx * tx + ty * ty <= HEAT_RADIUS * HEAT_RADIUS)
        {
            int idx = y * width + x;
            u[idx] += HEAT_SOURCE;
        }
    }
}

__global__ void add_heat_kernel_3d(float *u, int width, int height, int depth, int cx, int cy, int cz)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x - HEAT_RADIUS;
    int ty = blockIdx.y * blockDim.y + threadIdx.y - HEAT_RADIUS;
    int tz = blockIdx.z * blockDim.z + threadIdx.z - HEAT_RADIUS;

    int x = cx + tx;
    int y = cy + ty;
    int z = cz + tz;

    if (x >= 0 && x < width && y >= 0 && y < height && z >=0 && z < depth)
    {
        if (tx * tx + ty * ty + tz * tz <= HEAT_RADIUS * HEAT_RADIUS)
        {
            int idx = IDX_3D(x, y, z, width, height);
            u[idx] += HEAT_SOURCE;
        }
    }
}