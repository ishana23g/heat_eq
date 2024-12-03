// File: cuda_heat_equation_generalized.cu

#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glu.h>
#include <string>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Simulation settings
#define WIDTH 1000
#define HEIGHT 1000
#define DEPTH 100
#define HEAT_SOURCE 5.0f
#define DX 1.0f
#define DY 1.0f
#define DZ 1.0f

#define DX2 (DX * DX)
#define DY2 (DY * DY)
#define DZ2 (DZ * DZ)

#define HEAT_RADIUS 5
#define DIFFUSIVITY 1.0f

#define TIME_STEP (DX2 * DY2 * DZ2 / (2 * DIFFUSIVITY * (DX2 + DY2 + DZ2)))

// CUDA block size
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_Z 16

// Host variables
GLuint pbo;
struct cudaGraphicsResource *cuda_pbo_resource;
GLFWwindow *window;

// Device variables
float *d_u0, *d_u1;
uchar4 *d_output;

// Global variables for FPS calculation
double lastTime = 0.0;
int nbFrames = 0;
double fps = 0.0;

// Mouse state
bool is_mouse_pressed = false;

// Simulation modes
enum SimulationMode
{
    MODE_1D,
    MODE_2D,
    MODE_3D
};
SimulationMode simulation_mode = MODE_3D;

#define SCALE_FACTOR_3D 0.8f

// Boundary conditions
enum BoundaryCondition
{
    DIRICHLET,
    NEUMANN
};
BoundaryCondition boundary_condition = DIRICHLET;

// Debug Mode - For Profiling
bool debug_mode = false;
int MAX_TIME_STEPS = 100;
int PERCENT_ADD_HEAT_CHANCE = 40;

// Function prototypes
void init_simulation();
void init_opengl();
void update_sim_render();
void reset_simulation();
void keyboard_callback(GLFWwindow *window, int key, int scancode, int action, int mods);
void cursor_position_callback(GLFWwindow *window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
void add_heat_launcher(double xpos, double ypos);

// CUDA kernels
__global__ void heat_kernel_1d_fused(float *u0, float *u1, uchar4 *output,
                                     int width, float dt, float dx2, float a,
                                     BoundaryCondition boundary_condition);
__global__ void heat_kernel_2d_fused(float *u0, float *u1, uchar4 *output,
                                     int width, int height, float dt,
                                     float dx2, float dy2, float a,
                                     BoundaryCondition boundary_condition);
__global__ void heat_kernel_3d_fused(float *u0, float *u1, uchar4 *output,
                                     int width, int height, int depth,
                                     float dt,
                                     float dx2, float dy2, float dz2,
                                     float a, BoundaryCondition boundary_condition);

__global__ void add_heat_kernel_1d(float *u, int width, int x);
__global__ void add_heat_kernel_2d(float *u, int width, int height, int cx, int cy);
__global__ void add_heat_kernel_3d(float *u, int width, int height, int depth, int cx, int cy, int cz);

// Indexing macro
__device__ int IDX_2D(int x, int y, int width) {
    return (y * width) + x;
}

__device__ int IDX_3D(int x, int y, int z, int width, int height) {
    return z * width * height + y * width + x;
}

// Color Functions
__device__ void gradient_scaling(float heat_value, uchar4* out_color, SimulationMode mode);

// Clamp function
#define HEAT_MAX_CLAMP 1.0f
#define HEAT_MIN_CLAMP 0.0f
#define clamp(x) (x < HEAT_MIN_CLAMP ? HEAT_MIN_CLAMP : (x > HEAT_MAX_CLAMP ? HEAT_MAX_CLAMP : x))

__device__ void gradient_scaling(float heat_value, uchar4* out_color, SimulationMode mode)
{
    // Gradient Set Up:
    float t = clamp(heat_value / HEAT_SOURCE);

    // Default alpha
    unsigned char a = static_cast<unsigned char> (255.0f);

    // Adjust alpha for 3D mode
    if (mode == MODE_3D) {
        a = static_cast<unsigned char>(255.0f / DEPTH * 4.0f);
    }
#if 1
    // liner interpolation between rgb(0, 0, 255) and rgb(255, 0, 0)
    // const float R_low = 0.0f;
    const float R_high = 255.0f;
    // const float G_low = 0.0f;
    // const float G_high = 0.0f;
    const float B_low = 255.0f;
    const float B_high = 0.0f;
    
    // Perform linear interpolation
    unsigned char r = static_cast<unsigned char>(t * R_high);
    unsigned char g = static_cast<unsigned char>(0);
    unsigned char b = static_cast<unsigned char>(B_low + t * (B_high - B_low));

    // Write the result to the output pointer   
    *out_color = make_uchar4(r, g, b, a);
#else
    // liner interpolation between rgb(232, 251, 90) and rgb(5, 34, 51)
    const float R_low  = 232.0f;
    const float R_high = 5.0f;
    const float G_low  = 251.0f;
    const float G_high = 34.0f;
    const float B_low  = 90.0f;
    const float B_high = 51.0f;
    
    // Perform linear interpolation
    unsigned char r = static_cast<unsigned char>(R_low + t * (R_high - R_low));
    unsigned char g = static_cast<unsigned char>(G_low + t * (G_high - G_low));
    unsigned char b = static_cast<unsigned char>(B_low + t * (B_high - B_low));

    // Write the result to the output pointer
    *out_color = make_uchar4(r, g, b, a);
#endif
    // Could try: 
    // Define a color gradient MAGMA
    // magma_colormap = [
    //     [252, 253, 191],
    //     [254, 176, 120],
    //     [241, 96, 93],
    //     [183, 55, 121],
    //     [114, 31, 129],
    //     [44, 17, 95],
    //     [0, 0, 4]
    // ]
    // https://waldyrious.net/viridis-palette-generator/
    // However, this hard to convert to a linear interpolation
}

// CUDA kernel implementations

// Heat kernel for 1D simulation
/**
 * @brief Heat kernel for 1D simulation
 * Does the simulation and saving the output color values
 *
 * @param[in]   u0                      The heat values at the current time step (t)
 * @param[out]  u1                      The heat values at the next time step (t + dt)
 * @param[out]  output                  The output color values
 * @param[in]   width                   The width of the simulation area
 * @param[in]   dt                      The time step
 * @param[in]   dx2                     The square of the x-axis step size
 * @param[in]   a                       The diffusivity constant
 * @param[in]   boundary_condition      The boundary condition
 */
__global__ void heat_kernel_1d_fused(float *u0, float *u1, uchar4 *output,
                                     int width, float dt, float dx2, float a,
                                     BoundaryCondition boundary_condition)
{
    extern __shared__ float s_u[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int s_x = threadIdx.x + 1;

    if (x < width) {
        // Load data into shared memory
        // Central square
        s_u[s_x] = u0[x];

        // Load borders
        if (threadIdx.x == 0 && x > 0)
            s_u[0] = u0[blockIdx.x * blockDim.x - 1];
        // Right border
        if (threadIdx.x == blockDim.x - 1 && x < width - 1)
            s_u[BLOCK_SIZE_X + 1] = u0[(blockIdx.x + 1) * blockDim.x];

        // Load corners 
        // In this case it would be the left and right absolute borders
        if (threadIdx.x == 0 && x > 0)
            s_u[0] = u0[x - 1];
        if (threadIdx.x == blockDim.x - 1 && x < width - 1)
            s_u[BLOCK_SIZE_X + 1] = u0[x + 1];

        // Make sure all the data is loaded before computing
        __syncthreads();

        if (x > 0 && x < width - 1) {

            float u_center = s_u[s_x];
            float u_left = s_u[s_x - 1];
            float u_right = s_u[s_x + 1];

            float laplacian = (u_left - 2 * u_center + u_right) / dx2;

            u1[x] = u_center + a * dt * laplacian;

        }
        else if (x == 0 || x == width - 1)
        // (x < width)
        {
            switch (boundary_condition)
            {
            case DIRICHLET:
                u1[x] = 0.0f;
                break;
            case NEUMANN:
                // Modified Neumann boundary
                if (x == 0)
                    u1[x] = u1[x + 1] + HEAT_SOURCE * (dx2); // Left boundary
                else if (x == width - 1)
                    u1[x] = u1[x - 1] + HEAT_SOURCE * (dx2); // Right boundary
                break;
            }
        }
        
        gradient_scaling(u1[x], &output[x], MODE_1D);
    }
}


// Add heat kernel for 1D simulation
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



/**
 * @brief Fused both the simulate step and the color output into a single kernel
 * Also using shared memory to reduce global memory access
 *
 * @param u0         The heat values at the current time step (t)
 * @param u1         The heat values at the next time step (t + dt)
 * @param output     The output color values
 * @param width      The width of the simulation area
 * @param height     The height of the simulation area
 * @param dt         The time step
 * @param dx2        The square of the x-axis step size
 * @param dy2        The square of the y-axis step size
 * @param a          The diffusivity constant
 * @param boundary_condition The boundary condition
 */
__global__ void heat_kernel_2d_fused(float *u0, float *u1, uchar4 *output,
                                     int width, int height, float dt,
                                     float dx2, float dy2, float a,
                                     BoundaryCondition boundary_condition)
{
    // save a blocks size worth of data into shared memory
    // + 2 for the border on left and right sides both x and y
    extern __shared__ float s_u[];

    int shared_bs_y = BLOCK_SIZE_Y + 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // put into shared memory offset by 1 for the border
    int s_x = threadIdx.x + 1;
    int s_y = threadIdx.y + 1;

    if (x < width && y < height) {
        // Load data into shared memory
        // Load central square
        s_u[IDX_2D(s_x, s_y, shared_bs_y)] = u0[IDX_2D(x, y, width)];

        // Load borders
        if (threadIdx.x == 0 && x > 0)
            s_u[IDX_2D(0, s_y, shared_bs_y)] = u0[IDX_2D(x - 1, y, width)];
        if (threadIdx.x == blockDim.x - 1 && x < width - 1)
            s_u[IDX_2D(BLOCK_SIZE_X + 1, s_y, shared_bs_y)] = u0[IDX_2D(x + 1, y, width)];
        if (threadIdx.y == 0 && y > 0)
            s_u[IDX_2D(s_x, 0, shared_bs_y)] = u0[IDX_2D(x, y - 1, width)];
        if (threadIdx.y == blockDim.y - 1 && y < height - 1)
            s_u[IDX_2D(s_x, BLOCK_SIZE_Y + 1, shared_bs_y)] = u0[IDX_2D(x, y + 1, width)];

        // NOTE that from the CUDA Heat Equation implementation,
        //  the corners are not loaded into shared memory for some reason
        // they say that it isn't necessary, but I got errors. And GPT helped 
        // me figure out that it was because the corners were not loaded into shared memory

        // Load corners
        if (threadIdx.x == 0 && threadIdx.y == 0 && x > 0 && y > 0)
            s_u[IDX_2D(0, 0, shared_bs_y)] = u0[IDX_2D(x - 1, y - 1, width)];
        if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && x > 0 && y < height - 1)
            s_u[IDX_2D(0, BLOCK_SIZE_Y + 1, shared_bs_y)] = u0[IDX_2D(x - 1, y + 1, width)];
        if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && x < width - 1 && y > 0)
            s_u[IDX_2D(BLOCK_SIZE_X + 1, 0, shared_bs_y)] = u0[IDX_2D(x + 1, y - 1, width)];
        if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && x < width - 1 && y < height - 1)
            s_u[IDX_2D(BLOCK_SIZE_X + 1, BLOCK_SIZE_Y + 1, shared_bs_y)] = u0[IDX_2D(x + 1, y + 1, width)];

        // Ensure all threads have loaded their data
        __syncthreads();

        int idx = IDX_2D(x, y, width);

        if (x > 0 && x < width - 1 &&
            y > 0 && y < height - 1) {
            float u_center = s_u[IDX_2D(s_x, s_y, shared_bs_y)];
            float u_left = s_u[IDX_2D(s_x - 1, s_y, shared_bs_y)];
            float u_right = s_u[IDX_2D(s_x + 1, s_y, shared_bs_y)];
            float u_down = s_u[IDX_2D(s_x, s_y - 1, shared_bs_y)];
            float u_up = s_u[IDX_2D(s_x, s_y + 1, shared_bs_y)];

            float laplacian = (u_left - 2 * u_center + u_right ) / dx2 +
                            (u_up - 2 * u_center+ u_down ) / dy2;

            u1[idx] = u_center + a * dt * laplacian;
        }    
        // this part is using global memory
        else if (x == 0 || x == width - 1 || 
            y == 0 || y == height - 1) {
            switch (boundary_condition)
            {
            case DIRICHLET:
                u1[idx] = 0.0f;
                break;
            case NEUMANN:
                // Left boundary
                if (x == 0)
                    u1[idx] = u1[IDX_2D(x + 1, y, width)] + HEAT_SOURCE * dx2; 
                // Right boundary
                else if (x == width - 1)
                    u1[idx] = u1[IDX_2D(x - 1, y, width)] + HEAT_SOURCE * dx2; 
                // Top boundary
                else if (y == 0)
                    u1[idx] = u1[IDX_2D(x, y + 1, width)] + HEAT_SOURCE * dy2; 
                // Bottom boundary
                else if (y == height - 1)
                    u1[idx] = u1[IDX_2D(x, y - 1, width)] + HEAT_SOURCE * dy2; 
                break;
            }
        }

        gradient_scaling(u1[x], &output[x], MODE_2D);
    }
}

// Add heat kernel for 2D simulation
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

// Heat kernel for 3D simulation
/**
 * @brief Fused both the simulate step and the color output into a single kernel
 * Also using shared memory to reduce global memory access
 *
 * @param u0         The heat values at the current time step (t)
 * @param u1         The heat values at the next time step (t + dt)
 * @param output     The output color values
 * @param width      The width of the simulation area
 * @param height     The height of the simulation area
 * @param depth      The depth of the simulation area
 * @param dt         The time step
 * @param dx2        The square of the x-axis step size
 * @param dy2        The square of the y-axis step size
 * @param dz2        The square of the z-axis step size
 * @param a          The diffusivity constant
 * @param boundary_condition The boundary condition
 */
// __global__ void heat_kernel_3d_fused(float *u0, float *u1, uchar4 *output,
//                                      int width, int height, int depth,
//                                      float dt,
//                                      float dx2, float dy2, float dz2,
//                                      float a, BoundaryCondition boundary_condition)
// {
//     extern __shared__ float s_u[];

//     // save a blocks size worth of data into shared memory
//     // + 2 for the border on left and right sides both x and y
//     // __shared__ float s_u[(BLOCK_SIZE_X + 2) * (BLOCK_SIZE_Y + 2) * (BLOCK_SIZE_Z + 2)];

//     int shared_bs_y = BLOCK_SIZE_Y + 2;
//     int shared_bs_z = BLOCK_SIZE_Z + 2;
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int z = blockIdx.z * blockDim.z + threadIdx.z;

//     // put into shared memory offset by 1 for the border
//     int s_x = threadIdx.x + 1;
//     int s_y = threadIdx.y + 1;
//     int s_z = threadIdx.z + 1;

//     if (x < width && y < height && z < depth) {
//         // Load central square
//         s_u[IDX_3D(s_x, s_y, s_z, shared_bs_y, shared_bs_z)] = u0[IDX_3D(x, y, z, width, height)];

//         // Load borders (edges) into shared memory
//         if (threadIdx.x == 0 && x > 0)
//             s_u[IDX_3D(0, s_y, s_z, shared_bs_y, shared_bs_z)] = u0[IDX_3D(x - 1, y, z, width, height)];
//         if (threadIdx.x == blockDim.x - 1 && x < width - 1)
//             s_u[IDX_3D(BLOCK_SIZE_X + 1, s_y, s_z, shared_bs_y, shared_bs_z)] = u0[IDX_3D(x + 1, y, z, width, height)];
//         if (threadIdx.y == 0 && y > 0)
//             s_u[IDX_3D(s_x, 0, s_z, shared_bs_y, shared_bs_z)] = u0[IDX_3D(x, y - 1, z, width, height)];
//         if (threadIdx.y == blockDim.y - 1 && y < height - 1)
//             s_u[IDX_3D(s_x, BLOCK_SIZE_Y + 1, s_z, shared_bs_y, shared_bs_z)] = u0[IDX_3D(x, y + 1, z, width, height)];
//         if (threadIdx.z == 0 && z > 0)
//             s_u[IDX_3D(s_x, s_y, 0, shared_bs_y, shared_bs_z)] = u0[IDX_3D(x, y, z - 1, width, height)];
//         if (threadIdx.z == blockDim.z - 1 && z < depth - 1)
//             s_u[IDX_3D(s_x, s_y, BLOCK_SIZE_Z + 1, shared_bs_y, shared_bs_z)] = u0[IDX_3D(x, y, z + 1, width, height)];
        
//         // Load corners (vertices) into shared memory
//         if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && x > 0 && y > 0 && z > 0)
//             s_u[IDX_3D(0, 0, 0, shared_bs_y, shared_bs_z)] = u0[IDX_3D(x - 1, y - 1, z - 1, width, height)];
//         if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == blockDim.z - 1 && x > 0 && y > 0 && z < depth - 1)
//             s_u[IDX_3D(0, 0, BLOCK_SIZE_Z + 1, shared_bs_y, shared_bs_z)] = u0[IDX_3D(x - 1, y - 1, z + 1, width, height)];
//         if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && threadIdx.z == 0 && x > 0 && y < height - 1 && z > 0)
//             s_u[IDX_3D(0, BLOCK_SIZE_Y + 1, 0, shared_bs_y, shared_bs_z)] = u0[IDX_3D(x - 1, y + 1, z - 1, width, height)];
//         if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && threadIdx.z == blockDim.z - 1 && x > 0 && y < height - 1 && z < depth - 1)
//             s_u[IDX_3D(0, BLOCK_SIZE_Y + 1, BLOCK_SIZE_Z + 1, shared_bs_y, shared_bs_z)] = u0[IDX_3D(x - 1, y + 1, z + 1, width, height)];
//         if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && threadIdx.z == 0 && x < width - 1 && y > 0 && z > 0)
//             s_u[IDX_3D(BLOCK_SIZE_X + 1, 0, 0, shared_bs_y, shared_bs_z)] = u0[IDX_3D(x + 1, y - 1, z - 1, width, height)];
//         if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && threadIdx.z == blockDim.z - 1 && x < width - 1 && y > 0 && z < depth - 1)
//             s_u[IDX_3D(BLOCK_SIZE_X + 1, 0, BLOCK_SIZE_Z + 1, shared_bs_y, shared_bs_z)] = u0[IDX_3D(x + 1, y - 1, z + 1, width, height)];
//         if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && threadIdx.z == 0 && x < width - 1 && y < height - 1 && z > 0)
//             s_u[IDX_3D(BLOCK_SIZE_X + 1, BLOCK_SIZE_Y + 1, 0, shared_bs_y, shared_bs_z)] = u0[IDX_3D(x + 1, y + 1, z - 1, width, height)];
//         if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && threadIdx.z == blockDim.z - 1 && x < width - 1 && y < height - 1 && z < depth - 1)
//             s_u[IDX_3D(BLOCK_SIZE_X + 1, BLOCK_SIZE_Y + 1, BLOCK_SIZE_Z + 1, shared_bs_y, shared_bs_z)] = u0[IDX_3D(x + 1, y + 1, z + 1, width, height)];
        
//         // Ensure all threads have loaded their data
//         __syncthreads();

//         int idx = IDX_3D(x, y, z, width, height);

//         if (x > 0 && x < width - 1 &&
//             y > 0 && y < height - 1 &&
//             z > 0 && z < depth - 1) {

//             float u_center = s_u[IDX_3D(s_x, s_y, s_z, shared_bs_y, shared_bs_z)];
//             float u_left = s_u[IDX_3D(s_x - 1, s_y, s_z, shared_bs_y, shared_bs_z)];
//             float u_right = s_u[IDX_3D(s_x + 1, s_y, s_z, shared_bs_y, shared_bs_z)];
//             float u_down = s_u[IDX_3D(s_x, s_y - 1, s_z, shared_bs_y, shared_bs_z)];
//             float u_up = s_u[IDX_3D(s_x, s_y + 1, s_z, shared_bs_y, shared_bs_z)];
//             float u_back = s_u[IDX_3D(s_x, s_y, s_z - 1, shared_bs_y, shared_bs_z)];
//             float u_front = s_u[IDX_3D(s_x, s_y, s_z + 1, shared_bs_y, shared_bs_z)];

//             float laplacian = (u_left - 2 * u_center + u_right ) / dx2 +
//                             (u_up - 2 * u_center+ u_down ) / dy2 +
//                             (u_front - 2 * u_center + u_back) / dz2;

//             u1[idx] = u_center + a * dt * laplacian;
//         }    
//         // this part is using global memory
//         else if (x == 0 || x == width - 1 || y == 0 || y == height - 1 || z == 0 || z == depth - 1)
//         {
//             switch (boundary_condition)
//             {
//             case DIRICHLET:
//                 u1[idx] = 0.0f;
//                 break;
//             case NEUMANN:
//                 // Left boundary
//                 if (x == 0)
//                     u1[idx] = u0[IDX_3D(x + 1, y, z, width, height)] + HEAT_SOURCE * dx2;
//                 // Right boundary
//                 else if (x == width - 1)
//                     u1[idx] = u0[IDX_3D(x - 1, y, z, width, height)] + HEAT_SOURCE * dx2;
//                 // Top boundary
//                 else if (y == 0)
//                     u1[idx] = u0[IDX_3D(x, y + 1, z, width, height)] + HEAT_SOURCE * dy2;
//                 // Bottom boundary
//                 else if (y == height - 1)
//                     u1[idx] = u0[IDX_3D(x, y - 1, z, width, height)] + HEAT_SOURCE * dy2;
//                 // Front boundary
//                 else if (z == 0)
//                     u1[idx] = u0[IDX_3D(x, y, z + 1, width, height)] + HEAT_SOURCE * dz2;
//                 // Back boundary
//                 else if (z == depth - 1)
//                     u1[idx] = u0[IDX_3D(x, y, z - 1, width, height)] + HEAT_SOURCE * dz2;
//                 break;
//             }
//         }

//         gradient_scaling(u1[x], &output[x], MODE_3D);
//     }
// }

__global__ void heat_kernel_3d_fused(float* u0, float* u1, uchar4* output,
                                     int width, int height, int depth,
                                     float dt, float dx2, float dy2, float dz2,
                                     float a, BoundaryCondition boundary_condition) {

    extern __shared__ float slice[];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    int stride = width * height;
    int i2d = IDX_2D(ix, iy, width);
    int o2d = 0;
    bool compute_if = ix > 0 && ix < (width - 1) && iy > 0 && iy < (height - 1);

    float behind;
    float current = u0[i2d]; 
    o2d = i2d; 
    i2d += stride;
    float infront = u0[i2d]; 
    i2d += stride;

    for (int i = 1; i < depth - 1; i++) {

        // if i == 1 || i == depth - 2 (we need to update the boundaries)
        // it is the same as z == 0 || z == depth - 1
        if (i == 1 || i == depth - 2){
            switch (boundary_condition)
            {
            case DIRICHLET:
                u1[o2d] = 0.0f;
                break;
            case NEUMANN:
                // Left boundary
                if (i == 1)
                    u1[o2d] = u0[o2d + stride] + HEAT_SOURCE * dz2;
                // Right boundary
                else if (i == depth - 2)
                    u1[o2d] = u0[o2d - stride] + HEAT_SOURCE * dz2;
                break;
            }
        }

        // These go in registers:
        behind = current;
        current = infront;
        infront = u0[i2d];

        i2d += stride;
        o2d += stride;
        __syncthreads();

        // Shared memory

        if (compute_if) {
            if (threadIdx.x == 0) { // Halo left
                slice[IDX_2D(ty, tx - 1, BLOCK_SIZE_X + 2)] = u0[o2d - 1];
            }

            if (threadIdx.x == BLOCK_SIZE_X - 1) { // Halo right
                slice[IDX_2D(ty, tx + 1, BLOCK_SIZE_X + 2)] = u0[o2d + 1];
            }

            if (threadIdx.y == 0) { // Halo bottom
                slice[IDX_2D(ty - 1, tx, BLOCK_SIZE_X + 2)] = u0[o2d - width];
            }

            if (threadIdx.y == BLOCK_SIZE_Y - 1) { // Halo top
                slice[IDX_2D(ty + 1, tx, BLOCK_SIZE_X + 2)] = u0[o2d + width];
            }
        }

        __syncthreads();

        slice[IDX_2D(ty, tx, BLOCK_SIZE_X + 2)] = current;

        __syncthreads();

        if (compute_if) {
            u1[o2d] = current + (a * dt) * (
                (slice[IDX_2D(ty, tx - 1, BLOCK_SIZE_X + 2)] - 
                    2 * current + 
                    slice[IDX_2D(ty, tx + 1, BLOCK_SIZE_X + 2)]) / dx2 +
                (slice[IDX_2D(ty - 1, tx, BLOCK_SIZE_X + 2)] - 
                    2 * current + 
                    slice[IDX_2D(ty + 1, tx, BLOCK_SIZE_X + 2)]) / dy2 +
                (behind - 2 * current + infront) / dz2);
        }

        __syncthreads();

        if (ix == 0 || ix == width - 1 || 
            iy == 0 || iy == height - 1) {
            switch (boundary_condition)
            {
            case DIRICHLET:
                u1[o2d] = 0.0f;
                break;
            case NEUMANN:
                // Left boundary
                if (ix == 0)
                    u1[o2d] = u0[o2d + 1] + HEAT_SOURCE * dx2;                        
                // Right boundary
                else if (ix == width - 1)
                    u1[o2d] = u0[o2d - 1] + HEAT_SOURCE * dx2;
                // Top boundary
                else if (iy == 0)
                    u1[o2d] = u0[o2d + width] + HEAT_SOURCE * dy2;
                // Bottom boundary
                else if (iy == height - 1)
                    u1[o2d] = u0[o2d - width] + HEAT_SOURCE * dy2;
                break;
            }
        }
    }
}


__global__ void add_heat_kernel_3d(float *u, int width, int height, int depth, int cx, int cy, int cz) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x - HEAT_RADIUS;
    int ty = blockIdx.y * blockDim.y + threadIdx.y - HEAT_RADIUS;
    int tz = blockIdx.z * blockDim.z + threadIdx.z - HEAT_RADIUS;

    int x = cx + tx;
    int y = cy + ty;
    int z = cz + tz;

    if (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth) {
        if (tx * tx + ty * ty + tz * tz <= HEAT_RADIUS * HEAT_RADIUS) {
            u[IDX_3D(x, y, z, width, height)] += HEAT_SOURCE;
        }
    }
}


// Initialize the simulation
void init_simulation()
{
    size_t size;
    if (simulation_mode == MODE_1D)
    {
        size = WIDTH * sizeof(float);
    }
    else if (simulation_mode == MODE_2D)
    {
        size = WIDTH * HEIGHT * sizeof(float);
    }
    else if (simulation_mode == MODE_3D)
    {
        size = WIDTH * HEIGHT * DEPTH * sizeof(float);
    } 
    gpuErrchk(cudaMalloc((void **)&d_u0, size));
    gpuErrchk(cudaMalloc((void **)&d_u1, size));
    
    gpuErrchk(cudaMemset(d_u0, 0, size));
    gpuErrchk(cudaMemset(d_u1, 0, size));
}

// Initialize OpenGL
void init_opengl()
{
    if (!glfwInit())
    {
        printf("Failed to initialize GLFW\n");
        exit(-1);
    }

    int window_width = WIDTH;
    int window_height = (simulation_mode == MODE_1D) ? 100 : HEIGHT;
    // FPS calculation
    double currentTime = glfwGetTime();
    nbFrames++;
    char title[256];
    sprintf(title, "CUDA Heat Equation - Width: %d Height: %d FPS: NA", WIDTH, HEIGHT);
    if (currentTime - lastTime >= 1.0)
    {
        fps = double(nbFrames) / (currentTime - lastTime);
        nbFrames = 0;
        lastTime += 1.0;

        // Update window title
        char title[256];
        sprintf(title, "CUDA Heat Equation - Width: %d Height: %d FPS: %.2f", WIDTH, HEIGHT, fps);
    }

    window = glfwCreateWindow(window_width, window_height, title, NULL, NULL);
    if (!window)
    {
        printf("Failed to create window\n");
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(window);

    glewInit();

    // Create PBO
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    size_t pbo_size;
    if (simulation_mode == MODE_1D) {
        pbo_size = WIDTH * sizeof(uchar4);
    } else if (simulation_mode == MODE_2D) {
        pbo_size = WIDTH * HEIGHT * sizeof(uchar4);
    } else if (simulation_mode == MODE_3D) {
        pbo_size = WIDTH * HEIGHT * DEPTH * sizeof(uchar4);
    }
    glBufferData(GL_PIXEL_UNPACK_BUFFER, pbo_size, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Register PBO with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

// Render simulation
void update_sim_render()
{

    // Map PBO
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void **)&d_output, &size, cuda_pbo_resource);

    // float dx2 = DX * DX;
    // float dy2 = DY * DY;
    // float dz2 = DZ * DZ;

    if (simulation_mode == MODE_1D)
    {
        // if we just did BLOCK_SIZE_X, we would only be able to do 16 threads
        // that is too few, so we need to square it
        dim3 blockSize(BLOCK_SIZE_X * BLOCK_SIZE_X * 4);
        dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x);
        int sharedMemBytes = (blockSize.x+ 2) * sizeof(float);
        // print out the block size and grid size
        // printf("1D-kernel threads per block: %d\n", blockSize.x);
        // printf("1D-kernel threads per dimension:: %d\n", gridSize.x);
        // printf("1D-kernel shared memory: %d bytes \n", sharedMemBytes);

        heat_kernel_1d_fused<<<gridSize, blockSize, sharedMemBytes>>>(d_u0, d_u1, d_output, 
            WIDTH, TIME_STEP, 
            DX * DX, 
            DIFFUSIVITY, boundary_condition);
    }
    else if (simulation_mode == MODE_2D)
    {
        dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                      (HEIGHT + blockSize.y - 1) / blockSize.y);
        int sharedMemBytes = (blockSize.x + 2) * (blockSize.y + 2) * sizeof(float);

        // prnt out the block size and grid size
        // printf("2D-kernel threads per block: %d\n", blockSize.x * blockSize.y);
        // printf("2D-kernel threads per dimension: %d, %d\n", gridSize.x, gridSize.y);
        // printf("2D-kernel shared memory: %d bytes \n", sharedMemBytes);

        heat_kernel_2d_fused<<<gridSize, blockSize, sharedMemBytes>>>(d_u0, d_u1, d_output, 
            WIDTH, HEIGHT, TIME_STEP, 
            DX * DX, DY * DY, 
            DIFFUSIVITY, boundary_condition);
    }
    else if (simulation_mode == MODE_3D)
    {

        dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                      (HEIGHT + blockSize.y - 1) / blockSize.y);
        int sharedMemBytes = (blockSize.x + 2) * (blockSize.y + 2) * sizeof(float);


        // dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
        // dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
        //               (HEIGHT + blockSize.y - 1) / blockSize.y,
        //               (DEPTH + blockSize.z - 1) / blockSize.z);
        // int sharedMemBytes = (blockSize.x + 2) * (blockSize.y + 2) * (blockSize.z + 2) * sizeof(float);

        // print out the block size and grid size
        // printf("3D-kernel threads per block: %d\n", blockSize.x * blockSize.y * blockSize.z);
        // printf("3D-kernel threads per dimension: %d, %d, %d\n", gridSize.x, gridSize.y, gridSize.z);
        // printf("3D-kernel shared memory: %d bytes \n", sharedMemBytes);

        heat_kernel_3d_fused<<<gridSize, blockSize, sharedMemBytes>>>(d_u0, d_u1, d_output, 
            WIDTH, HEIGHT, DEPTH, TIME_STEP, 
            DX * DX, DY * DY, DZ * DZ,
            DIFFUSIVITY, boundary_condition);
    }

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Swap pointers
    float *temp = d_u0;
    d_u0 = d_u1;
    d_u1 = temp;

    // FPS calculation
    double currentTime = glfwGetTime();
    nbFrames++;
    if (currentTime - lastTime >= 1.0)
    {
        fps = double(nbFrames) / (currentTime - lastTime);
        nbFrames = 0;
        lastTime += 1.0;

        // Update window title
        char title[256];
        sprintf(title, "CUDA Heat Equation - Width: %d Height: %d FPS: %.2f", WIDTH, HEIGHT, fps);
        glfwSetWindowTitle(window, title);
    }

    // Unmap PBO
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

    // Draw pixels
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    if (simulation_mode == MODE_1D)
    {
        // HEIGHT HAS BEEN CHANGED TO 10
        glDrawPixels(WIDTH, 1, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    }
    else if (simulation_mode == MODE_2D)
    {
        glRasterPos2i(-1, -1);
        glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    }
    else if (simulation_mode == MODE_3D)
    {

        int window_width, window_height;
        glfwGetFramebufferSize(window, &window_width, &window_height);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Set up orthographic projection
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0, window_width, 0, window_height);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Starting position for the backmost layer
        float start_x = 0.0f;
        float start_y = window_height - (HEIGHT * SCALE_FACTOR_3D);

        float offset_x = start_x;
        float offset_y = start_y;

        // Set pixel zoom for scaling for all new slices
        glPixelZoom(SCALE_FACTOR_3D, SCALE_FACTOR_3D);
        
        // Draw each slice with offset
        for (int z = 0; z < DEPTH; ++z)
        {
            // Calculate position offset
            offset_x += (WIDTH * SCALE_FACTOR_3D) / DEPTH;
            offset_y -= (HEIGHT * SCALE_FACTOR_3D) / DEPTH;

            // Set raster position
            glRasterPos2f(offset_x, offset_y);

            // Draw the slice
            glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid *)(z * WIDTH * HEIGHT * sizeof(uchar4)));
        }

        // Reset pixel zoom
        glPixelZoom(1.0f, 1.0f);

        glDisable(GL_BLEND);
    }
    
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glfwSwapBuffers(window);
}

// Reset simulation
void reset_simulation()
{
    size_t size;
    if (simulation_mode == MODE_1D)
    {
        size = WIDTH * sizeof(float);
    }
    else if (simulation_mode == MODE_2D)
    {
        size = WIDTH * HEIGHT * sizeof(float);
    }
    else if (simulation_mode == MODE_3D)
    {
        size = WIDTH * HEIGHT * DEPTH * sizeof(float);
    }
    cudaMemset(d_u0, 0, size);
    cudaMemset(d_u1, 0, size);
}

// Keyboard callback
void keyboard_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_1:
            simulation_mode = MODE_1D;
            printf("Switched to 1D simulation\n");
            cudaGraphicsUnregisterResource(cuda_pbo_resource);
            glDeleteBuffers(1, &pbo);
            cudaFree(d_u0);
            cudaFree(d_u1);
            glfwDestroyWindow(window);
            init_opengl();
            init_simulation();

            glfwSetKeyCallback(window, keyboard_callback);
            glfwSetMouseButtonCallback(window, mouse_button_callback);
            glfwSetCursorPosCallback(window, cursor_position_callback);

            break;
        case GLFW_KEY_2:
            simulation_mode = MODE_2D;
            printf("Switched to 2D simulation\n");
            cudaGraphicsUnregisterResource(cuda_pbo_resource);
            glDeleteBuffers(1, &pbo);
            cudaFree(d_u0);
            cudaFree(d_u1);
            glfwDestroyWindow(window);
            init_opengl();
            init_simulation();

            glfwSetKeyCallback(window, keyboard_callback);
            glfwSetMouseButtonCallback(window, mouse_button_callback);
            glfwSetCursorPosCallback(window, cursor_position_callback);

            break;
        case GLFW_KEY_3:
            simulation_mode = MODE_3D;
            printf("Switched to 3D simulation\n");
            cudaGraphicsUnregisterResource(cuda_pbo_resource);
            glDeleteBuffers(1, &pbo);
            cudaFree(d_u0);
            cudaFree(d_u1);
            glfwDestroyWindow(window);
            init_opengl();
            init_simulation();

            glfwSetKeyCallback(window, keyboard_callback);
            glfwSetMouseButtonCallback(window, mouse_button_callback);
            glfwSetCursorPosCallback(window, cursor_position_callback);

            break;
        case GLFW_KEY_B:
            boundary_condition = (boundary_condition == DIRICHLET) ? NEUMANN : DIRICHLET;
            printf("Switched boundary condition to %s\n",
                   (boundary_condition == DIRICHLET) ? "Dirichlet" : "Neumann");
            break;
        case GLFW_KEY_R:
            reset_simulation();
            printf("Simulation reset\n");
            break;
        case GLFW_KEY_X:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            printf("Exiting simulation\n");

            // Cleanup
            glDeleteBuffers(1, &pbo);
            cudaFree(d_u0);
            cudaFree(d_u1);
            glfwDestroyWindow(window);
            glfwTerminate();
            exit(0);
            break;
        default:
            break;
        }
    }
}

void add_heat_launcher(double xpos, double ypos){
    int fb_width, fb_height;
    glfwGetFramebufferSize(window, &fb_width, &fb_height);

    if (simulation_mode == MODE_1D)
    {
        int x = (int)(xpos * WIDTH / fb_width);

        dim3 blockSize(256);
        dim3 gridSize((2 * HEAT_RADIUS + blockSize.x - 1) / blockSize.x);

        add_heat_kernel_1d<<<gridSize, blockSize>>>(d_u0, WIDTH, x);
    }
    else if (simulation_mode == MODE_2D)
    {
        int x = (int)(xpos * WIDTH / fb_width);
        int y = (int)((fb_height - ypos) * HEIGHT / fb_height); // Invert y-axis
        dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 gridSize((2 * HEAT_RADIUS + blockSize.x - 1) / blockSize.x,
                        (2 * HEAT_RADIUS + blockSize.y - 1) / blockSize.y);

        add_heat_kernel_2d<<<gridSize, blockSize>>>(d_u0, WIDTH, HEIGHT, x, y);
    }
    else if (simulation_mode == MODE_3D)
    {
        int x = (int)(xpos * WIDTH / fb_width);
        int y = (int)((fb_height - ypos) * HEIGHT / fb_height); // Invert y-axis
        // int z = DEPTH / 2; // Set z to the middle of the depth
        int z = rand() % DEPTH;
        dim3 blockSize(BLOCK_SIZE_X/2, BLOCK_SIZE_Y/2, BLOCK_SIZE_Z/2);
        dim3 gridSize((2 * HEAT_RADIUS + blockSize.x - 1) / blockSize.x,
                        (2 * HEAT_RADIUS + blockSize.y - 1) / blockSize.y,
                        (2 * HEAT_RADIUS + blockSize.z - 1) / blockSize.z);

        add_heat_kernel_3d<<<gridSize, blockSize>>>(d_u0, WIDTH, HEIGHT, DEPTH, x, y, z);
    }

    cudaDeviceSynchronize();
}

// Cursor position callback
void cursor_position_callback(GLFWwindow *window, double xpos, double ypos)
{
    if (is_mouse_pressed)
    {
        add_heat_launcher(xpos, ypos);
    }
}

// Mouse button callback
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            is_mouse_pressed = true;
            cursor_position_callback(window, 0, 0); // Trigger heat addition
        }
        else if (action == GLFW_RELEASE)
        {
            is_mouse_pressed = false;
        }
    }
}

// Print usage
void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("[options]:\n");
    printf("  -m <1d|2d|3d>         Simulation mode (default: 3d)\n");
    printf("  -b <D|N>              Boundary condition: D (Dirichlet), N (Neumann) (default: D)\n");
    printf("  -d [max_steps chance] Enable debug mode with optional max time steps (default: 100) and heat chance (default: 40)\n");
}

int main(int argc, char **argv) {
    int opt;
    while ((opt = getopt(argc, argv, "m:b:d::")) != -1) {
        switch (opt) {
            case 'm': // Simulation mode
                if (strcmp(optarg, "1d") == 0) {
                    simulation_mode = MODE_1D;
                } else if (strcmp(optarg, "2d") == 0) {
                    simulation_mode = MODE_2D;
                } else if (strcmp(optarg, "3d") == 0) {
                    simulation_mode = MODE_3D;
                } else {
                    fprintf(stderr, "Invalid simulation mode: %s\n", optarg);
                    print_usage(argv[0]);
                    return EXIT_FAILURE;
                }
                break;

            case 'b': // Boundary condition
                if (strcmp(optarg, "D") == 0) {
                    boundary_condition = DIRICHLET;
                } else if (strcmp(optarg, "N") == 0) {
                    boundary_condition = NEUMANN;
                } else {
                    fprintf(stderr, "Invalid boundary condition: %s\n", optarg);
                    print_usage(argv[0]);
                    return EXIT_FAILURE;
                }
                break;

            case 'd': // Debug mode
                debug_mode = true;
                if (optind < argc && argv[optind][0] != '-') {
                    MAX_TIME_STEPS = atoi(argv[optind++]);
                }
                if (optind < argc && argv[optind][0] != '-') {
                    PERCENT_ADD_HEAT_CHANCE = atoi(argv[optind++]);
                }

                // Validate debug parameters
                if (MAX_TIME_STEPS < 0) MAX_TIME_STEPS = 100;
                if (PERCENT_ADD_HEAT_CHANCE < 0 || PERCENT_ADD_HEAT_CHANCE > 100) {
                    PERCENT_ADD_HEAT_CHANCE = 40;
                }
                break;

            default: // Invalid option
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    // Print the parsed configuration
    printf("Simulation Mode: %s\n", simulation_mode == MODE_1D ? "1D" :
                                  simulation_mode == MODE_2D ? "2D" : "3D");
    printf("Boundary Condition: %s\n", boundary_condition == DIRICHLET ? "Dirichlet" : "Neumann");
    printf("Debug Mode: %s\n", debug_mode ? "Enabled" : "Disabled");
    if (debug_mode) {
        printf("  Max Time Steps: %d\n", MAX_TIME_STEPS);
        printf("  Heat Chance: %d%%\n", PERCENT_ADD_HEAT_CHANCE);
    }
    printf("====================================\n");

    init_opengl();
    init_simulation();

    glfwSetKeyCallback(window, keyboard_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);


    if (debug_mode)
    {
        // Debug mode - Add heat to the center of the simulation
        for (int i = 0; i < MAX_TIME_STEPS; i++)
        {
            // CANNOT MANUALLY ADD HEAT IN DEBUG MODE
            // NOR CAN MOUSE/KEYBOARD INPUT BE USED
            update_sim_render();

            // randomlly add heat to the simulation, not at all time steps
            if (rand() % 100 < PERCENT_ADD_HEAT_CHANCE)
            {
                int x = rand() % WIDTH;
                int y = rand() % HEIGHT;
                add_heat_launcher(x, y);
            }
        }
        return 0;
    }
    else
    {
        // Main loop
        while (window && !glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            update_sim_render();
        }
    }

    // Cleanup
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    cudaFree(d_u0);
    cudaFree(d_u1);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
