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
// Declare the texture globally or in an appropriate scope
GLuint volumeTexture;

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
void init_volume_texture();
void init();
void update_sim_render();
void reset_simulation();
void cleanup();
void print_config();
void keyboard_callback(GLFWwindow *window, int key, int scancode, int action, int mods);
void cursor_position_callback(GLFWwindow *window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
void add_heat_launcher(double xpos, double ypos);

// CUDA kernels
__global__ void heat_kernel_1d_sim(float *u0, float *u1,
                                     int width, float dt, float dx2, float a,
                                     BoundaryCondition boundary_condition);
__global__ void heat_kernel_2d_sim(float *u0, float *u1,
                                     int width, int height, float dt,
                                     float dx2, float dy2, float a,
                                     BoundaryCondition boundary_condition);
__global__ void heat_kernel_3d_sim(float *u0, float *u1,
                                     int width, int height, int depth,
                                     float dt,
                                     float dx2, float dy2, float dz2,
                                     float a, BoundaryCondition boundary_condition);
__global__ void heat_kernel_1d_color(float *u, uchar4 *output, int width);
__global__ void heat_kernel_2d_color(float *u, uchar4 *output, int width, int height);
__global__ void heat_kernel_3d_color(float *u, uchar4 *output, int width, int height, int depth);
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

/**
 * @brief Gradient scaling function
 * 
 * @param heat_value The heat value
 * @param out_color The output color
 * @param mode The simulation mode
 * 
 * There are two constant colors that are used to create a gradient
 * The gradient is created by interpolating between the two colors
 */
__device__ void gradient_scaling(float heat_value, uchar4* out_color, SimulationMode mode)
{

    // COLORS CAN BE SET HERE.
    // THE DEFAULT IS A BLUE TO YELLOW GRADIENT
    #if 1
    // liner interpolation between rgb(5, 34, 51) and rgb(232, 251, 90)
    const uchar4 LOW_COLOR = make_uchar4(5.0f, 34.0f, 51.0f, 255.0f);
    const uchar4 HIGH_COLOR = make_uchar4(232.0f, 251.0f, 90.0f, 255.0f);
    #else
    // liner interpolation between rgb(0, 0, 255) and rgb(255, 0, 0)
    const uchar4 LOW_COLOR = make_uchar4(0.0f, 0.0f, 255.0f, 255.0f);
    const uchar4 HIGH_COLOR = make_uchar4(255.0f, 0.0f, 0.0f, 255.0f);
    #endif

    // Gradient Set Up:
    float t = clamp(heat_value / HEAT_SOURCE);

    // Default alpha
    unsigned char a = static_cast<unsigned char> (255.0f);

    // Adjust alpha for 3D mode
    if (mode == MODE_3D) {
        a = static_cast<unsigned char>(255.0f / DEPTH * 4.0f);
    }

    *out_color = make_uchar4(
        LOW_COLOR.x + t * (HIGH_COLOR.x - LOW_COLOR.x),
        LOW_COLOR.y + t * (HIGH_COLOR.y - LOW_COLOR.y),
        LOW_COLOR.z + t * (HIGH_COLOR.z - LOW_COLOR.z),
        a
    );
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
            s_u[blockDim.x + 1] = u0[(blockIdx.x + 1) * blockDim.x];

        // Load corners 
        // In this case it would be the left and right absolute borders
        if (threadIdx.x == 0 && x > 0)
            s_u[0] = u0[x - 1];
        if (threadIdx.x == blockDim.x - 1 && x < width - 1)
            s_u[blockDim.x + 1] = u0[x + 1];

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


// 1D simulation kernel -----------------------------------------

/**
 * @brief Heat kernel for 1D simulation
 *
 * @param[in]   u0                      The heat values at the current time step (t)
 * @param[out]  u1                      The heat values at the next time step (t + dt)
 * @param[in]   width                   The width of the simulation area
 * @param[in]   dt                      The time step
 * @param[in]   dx2                     The square of the x-axis step size
 * @param[in]   a                       The diffusivity constant
 * @param[in]   boundary_condition      The boundary condition
 */
__global__ void heat_kernel_1d_sim(float *u0, float *u1,
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
            s_u[blockDim.x + 1] = u0[(blockIdx.x + 1) * blockDim.x];

        // Load corners 
        // In this case it would be the left and right absolute borders
        if (threadIdx.x == 0 && x > 0)
            s_u[0] = u0[x - 1];
        if (threadIdx.x == blockDim.x - 1 && x < width - 1)
            s_u[blockDim.x + 1] = u0[x + 1];

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
    }
}


// Heat kernel for 1D simulation
/**
 * @brief Just coloring the output
 *
 * @param[in]   u                       The heat values at the current time step (t)
 * @param[out]  output                  The output color values
 * @param[in]   width                   The width of the simulation area
 */
__global__ void heat_kernel_1d_color(float *u, uchar4 *output, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < width) {
        gradient_scaling(u[x], &output[x], MODE_1D);
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

// 2D simulation kernel -----------------------------------------

/**
 * @brief Heat kernel for 2D simulation
 *
 * @param u0         The heat values at the current time step (t)
 * @param u1         The heat values at the next time step (t + dt)
 * @param width      The width of the simulation area
 * @param height     The height of the simulation area
 * @param dt         The time step
 * @param dx2        The square of the x-axis step size
 * @param dy2        The square of the y-axis step size
 * @param a          The diffusivity constant
 * @param boundary_condition The boundary condition
 */
__global__ void heat_kernel_2d_sim(float *u0, float *u1,
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
                    u1[idx] = s_u[IDX_2D(s_x + 1, s_y, shared_bs_y)] + HEAT_SOURCE * dx2; 
                // Right boundary
                else if (x == width - 1)
                    u1[idx] = s_u[IDX_2D(s_x - 1, s_y, shared_bs_y)] + HEAT_SOURCE * dx2; 
                // Top boundary
                else if (y == 0)
                    u1[idx] = s_u[IDX_2D(s_x, s_y + 1, shared_bs_y)] + HEAT_SOURCE * dy2; 
                // Bottom boundary
                else if (y == height - 1)
                    u1[idx] = s_u[IDX_2D(s_x, s_y - 1, shared_bs_y)] + HEAT_SOURCE * dy2; 
                break;
            }
        }
    }
}

/**
 * @brief Just coloring the output for 2D simulation
 *
 * @param[in]   u                       The heat values at the current time step (t)
 * @param[out]  output                  The output color values
 * @param[in]   width                   The width of the simulation area
 * @param[in]   height                  The height of the simulation area
 */
__global__ void heat_kernel_2d_color(float *u, uchar4 *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = IDX_2D(x, y, width);
        gradient_scaling(u[idx], &output[idx], MODE_2D);
    }
}

/**
 * @brief Add heat kernel for 2D simulation
 * 
 * @param u         The heat values at the current time step (t)
 * @param width     The width of the simulation area
 * @param height    The height of the simulation area
 * @param cx        The x-coordinate of the center of the heat source
 * @param cy        The y-coordinate of the center of the heat source
 */
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

// 3D simulation kernel -----------------------------------------
/**
 * @brief Heat kernel for 3D simulation
 *
 * @param u0         The heat values at the current time step (t)
 * @param u1         The heat values at the next time step (t + dt)
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
__global__ void heat_kernel_3d_sim(float* u0, float* u1,
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



/**
 * @brief Just coloring the output for 3D simulation
 *
 * @param[in]   u                       The heat values at the current time step (t)
 * @param[out]  output                  The output color values
 * @param[in]   width                   The width of the simulation area
 * @param[in]   height                  The height of the simulation area
 * @param[in]   depth                   The depth of the simulation area
 */
__global__ void heat_kernel_3d_color(float *u, uchar4 *output, int width, int height, int depth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        int idx = IDX_3D(x, y, z, width, height);
        gradient_scaling(u[idx], &output[idx], MODE_3D);
    }
}

/**
 * @brief Add heat kernel for 3D simulation
 * 
 * @param u         The heat values at the current time step (t)
 * @param width     The width of the simulation area
 * @param height    The height of the simulation area
 * @param depth     The depth of the simulation area
 * @param cx        The x-coordinate of the center of the heat source
 * @param cy        The y-coordinate of the center of the heat source
 * @param cz        The z-coordinate of the center of the heat source
 */
__global__ void add_heat_kernel_3d(float *u, int width, int height, int depth, int cx, int cy, int cz) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x - (HEAT_RADIUS * 2);
    int ty = blockIdx.y * blockDim.y + threadIdx.y - (HEAT_RADIUS * 2);
    int tz = blockIdx.z * blockDim.z + threadIdx.z - (HEAT_RADIUS * 2);

    int x = cx + tx;
    int y = cy + ty;
    int z = cz + tz;

    if (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth) {
        if (tx * tx + ty * ty + tz * tz <= HEAT_RADIUS * HEAT_RADIUS * 4) {
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

    init_volume_texture();
}


void init_volume_texture() {
    glGenTextures(1, &volumeTexture);
    glBindTexture(GL_TEXTURE_3D, volumeTexture);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_3D, 0);
}

// Render simulation
void update_sim_render()
{
    // Map PBO
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void **)&d_output, &size, cuda_pbo_resource);

    const int MAX_SIM_STEPS = 10;
    
    for (int i=0; i<MAX_SIM_STEPS; i++){
        if (simulation_mode == MODE_1D)
        {
            // if we just did BLOCK_SIZE_X, we would only be able to do 16 threads
            // that is too few, so we need to square it
            dim3 blockSize(BLOCK_SIZE_X * BLOCK_SIZE_X * 2);
            dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x);
            int sharedMemBytes = (blockSize.x+ 2) * sizeof(float);
            // simulate for all steps
            heat_kernel_1d_sim<<<gridSize, blockSize, sharedMemBytes>>>(d_u0, d_u1,
                WIDTH, TIME_STEP, 
                DX * DX, 
                DIFFUSIVITY, boundary_condition);
            // color updated at last step
            if (i == MAX_SIM_STEPS - 1){   
                heat_kernel_1d_color<<<gridSize, blockSize>>>(d_u1, 
                    d_output, WIDTH);
            }
    
        }
        else if (simulation_mode == MODE_2D)
        {
            dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                        (HEIGHT + blockSize.y - 1) / blockSize.y);
            int sharedMemBytes = (blockSize.x + 2) * (blockSize.y + 2) * sizeof(float);

            heat_kernel_2d_sim<<<gridSize, blockSize, sharedMemBytes>>>(d_u0, d_u1,
                WIDTH, HEIGHT, TIME_STEP, 
                DX * DX, DY * DY, 
                DIFFUSIVITY, boundary_condition);
            
            if (i == MAX_SIM_STEPS - 1){
                heat_kernel_2d_color<<<gridSize, blockSize>>>(d_u1, 
                    d_output, WIDTH, HEIGHT);
            }
        }
        else if (simulation_mode == MODE_3D)
        {

            dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                        (HEIGHT + blockSize.y - 1) / blockSize.y);
            int sharedMemBytes = (blockSize.x + 2) * (blockSize.y + 2) * sizeof(float);

            heat_kernel_3d_sim<<<gridSize, blockSize, sharedMemBytes>>>(d_u0, d_u1,
                WIDTH, HEIGHT, DEPTH, TIME_STEP, 
                DX * DX, DY * DY, DZ * DZ,
                DIFFUSIVITY, boundary_condition);

            if (i == MAX_SIM_STEPS - 1){
                dim3 blockSize(BLOCK_SIZE_X/2, BLOCK_SIZE_Y/2, BLOCK_SIZE_Z/2);
                dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                            (HEIGHT + blockSize.y - 1) / blockSize.y,
                            (DEPTH + blockSize.z - 1) / blockSize.z);
                heat_kernel_3d_color<<<gridSize, blockSize>>>(d_u1, 
                    d_output, WIDTH, HEIGHT, DEPTH);
            }

        }
        gpuErrchk(cudaPeekAtLastError());

        // Swap pointers
        std::swap(d_u0, d_u1);
    }

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
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    
    if (simulation_mode == MODE_1D)
    {  
        glDisable(GL_DEPTH_TEST);
        for (int i = 0; i < 20; i++)
        {
            glRasterPos2f(-1.0f, -1.0f + i * 0.02f);
            glDrawPixels(WIDTH, 1, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        }
    }
    else if (simulation_mode == MODE_2D)
    {
        glDisable(GL_DEPTH_TEST);
        glRasterPos2i(-1, -1);
        glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    }
    else if (simulation_mode == MODE_3D)
    {   
        // Bind the volume texture
        glBindTexture(GL_TEXTURE_3D, volumeTexture);

        // Transfer data from PBO to the 3D texture
        glTexSubImage3D(
            GL_TEXTURE_3D,          // Target
            0,                      // Level
            0, 0, 0,                // Offset
            WIDTH, HEIGHT, DEPTH,   // Size
            GL_RGBA,                // Format
            GL_UNSIGNED_BYTE,       // Type
            0);                     // Offset in PBO

        // Render slices from back to front
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_TEXTURE_3D);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Set up the projection and view matrices
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45.0, (double)WIDTH / (double)HEIGHT, 0.1, 1000.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(
            0.0, 0.0, 2.0,  // Camera position
            0.0, 0.0, 0.0,  // Look at point
            0.0, 1.0, 0.0); // Up vector

        // Render slices from back to front
        glBegin(GL_QUADS);
        for (int i = 0; i < DEPTH; ++i)
        {
            float z = -1.0f + 2.0f * i / (DEPTH - 1);
            float texZ = (float)i / (DEPTH - 1);

            glTexCoord3f(0.0f, 0.0f, texZ); glVertex3f(-1.0f, -1.0f, z);
            glTexCoord3f(1.0f, 0.0f, texZ); glVertex3f( 1.0f, -1.0f, z);
            glTexCoord3f(1.0f, 1.0f, texZ); glVertex3f( 1.0f,  1.0f, z);
            glTexCoord3f(0.0f, 1.0f, texZ); glVertex3f(-1.0f,  1.0f, z);
        }
        glEnd();

        glDisable(GL_BLEND);
        glDisable(GL_TEXTURE_3D);
        glDisable(GL_DEPTH_TEST);

        // int window_width, window_height;
        // glfwGetFramebufferSize(window, &window_width, &window_height);

        // glEnable(GL_BLEND);
        // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // // Set up orthographic projection
        // glMatrixMode(GL_PROJECTION);
        // glLoadIdentity();
        // gluOrtho2D(0, window_width, 0, window_height);
        // glMatrixMode(GL_MODELVIEW);
        // glLoadIdentity();

        // // Starting position for the backmost layer
        // float start_x = 0.0f;
        // float start_y = window_height - (HEIGHT * SCALE_FACTOR_3D);

        // float offset_x = start_x;
        // float offset_y = start_y;

        // // Set pixel zoom for scaling for all new slices
        // glPixelZoom(SCALE_FACTOR_3D, SCALE_FACTOR_3D);
        
        // // Draw each slice with offset
        // for (int z = 0; z < DEPTH; ++z)
        // {
        //     // Calculate position offsetfused
        //     offset_x += (WIDTH * SCALE_FACTOR_3D) / DEPTH;
        //     offset_y -= (HEIGHT * SCALE_FACTOR_3D) / DEPTH;

        //     // Set raster position
        //     glRasterPos2f(offset_x, offset_y);

        //     // Draw the slice
        //     glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid *)(z * WIDTH * HEIGHT * sizeof(uchar4)));
        // }

        // // Reset pixel zoom
        // glPixelZoom(1.0f, 1.0f);

        // glDisable(GL_BLEND);
    }

    // Unbind PBO
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
            cleanup();
            init();
            break;
        case GLFW_KEY_2:
            simulation_mode = MODE_2D;
            printf("Switched to 2D simulation\n");
            cleanup();
            init();
            break;
        case GLFW_KEY_3:
            simulation_mode = MODE_3D;
            printf("Switched to 3D simulation\n");
            cleanup();
            init();
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

            cleanup();
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
        // Calculate x and y based on scaling factor
        int x = (int)(xpos * WIDTH / fb_width / SCALE_FACTOR_3D);
        int y = (int)((fb_height - ypos) * HEIGHT / fb_height / SCALE_FACTOR_3D); // Invert y-axis

        // Calculate z based on the cursor position and scaling factor
        float offset_x = xpos - (WIDTH * SCALE_FACTOR_3D) / DEPTH;
        float offset_y = (fb_height - ypos) - (HEIGHT * SCALE_FACTOR_3D) / DEPTH;
        int z = (int)((offset_x + offset_y) / ((WIDTH * SCALE_FACTOR_3D) / DEPTH));
        z = clamp(z / DEPTH - 1);
        
        dim3 blockSize(BLOCK_SIZE_X/2, BLOCK_SIZE_Y/2, BLOCK_SIZE_Z/2);
        dim3 gridSize((2 * HEAT_RADIUS + blockSize.x - 1) / blockSize.x,
                        (2 * HEAT_RADIUS + blockSize.y - 1) / blockSize.y,
                        (2 * HEAT_RADIUS + blockSize.z - 1) / blockSize.z);

        add_heat_kernel_3d<<<gridSize, blockSize>>>(d_u0, WIDTH, HEIGHT, DEPTH, x, y, z);
    }

    // cudaDeviceSynchronize();
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

// Print the parsed configuration
void print_config(){
    printf("====================================\n");
    printf("Simulation Mode: %s\n", simulation_mode == MODE_1D ? "1D" :
                                  simulation_mode == MODE_2D ? "2D" : "3D");
    printf("Boundary Condition: %s\n", boundary_condition == DIRICHLET ? "Dirichlet" : "Neumann");
    printf("Debug Mode: %s\n", debug_mode ? "Enabled" : "Disabled");
    if (debug_mode) {
        printf("  Max Time Steps: %d\n", MAX_TIME_STEPS);
        printf("  Heat Chance: %d%%\n", PERCENT_ADD_HEAT_CHANCE);
    }
    printf("====================================\n");
}

void init(){
    print_config();
    init_opengl();
    init_simulation();

    glfwSetKeyCallback(window, keyboard_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
}

void cleanup(){
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteTextures(1, &volumeTexture);
    glDeleteBuffers(1, &pbo);
    cudaFree(d_u0);
    cudaFree(d_u1);
    glfwDestroyWindow(window);
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

    init();

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
    cleanup();
    glfwTerminate();
    exit(0);
}
