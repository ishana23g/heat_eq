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
// h = \Delta x = \Delta y = \Delta z
#define H 1.0f
#define H2 (H * H)
#define H2_INV (1.0f / H2)
// #define DX (H)
// #define DY (H)
// #define DZ (H)

// #define DX2 (DX * DX)
// #define DY2 (DY * DY)
// #define DZ2 (DZ * DZ)

// #define DX2_INV (1.0f / DX2) 
// #define DY2_INV (1.0f / DY2)
// #define DZ2_INV (1.0f / DZ2)

#define HEAT_RADIUS 5
#define DIFFUSIVITY 1.0f

#define TIME_STEP (H2 * H2 * H2 / (2 * DIFFUSIVITY * (H2 + H2 + H2)))

// r = \frac{\Delta t \, \alpha}{h^2}
#define RATIO (TIME_STEP * DIFFUSIVITY / H)

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
__global__ void heat_kernel_1d_sim(float *u0, float *u1, BoundaryCondition boundary_condition);
__global__ void heat_kernel_2d_sim(float *u0, float *u1, BoundaryCondition boundary_condition);
__global__ void heat_kernel_3d_sim(float *u0, float *u1, BoundaryCondition boundary_condition);
__global__ void heat_kernel_1d_color(float *u, uchar4 *output);
__global__ void heat_kernel_2d_color(float *u, uchar4 *output);
__global__ void heat_kernel_3d_color(float *u, uchar4 *output);
__global__ void add_heat_kernel_1d(float *u, int width, int x);
__global__ void add_heat_kernel_2d(float *u, int width, int height, int cx, int cy);
__global__ void add_heat_kernel_3d(float *u, int width, int height, int depth, int cx, int cy, int cz);

// // Indexing macro
#define IDX_2D(x, y, width) ((y) * (width) + (x))
#define IDX_3D(x, y, z, width, height) ((z) * (width) * (height) + (y) * (width) + (x))

// __device__ int IDX_2D(int x, int y, int width) {
//     return (y * width) + x;
// }

// __device__ int IDX_3D(int x, int y, int z, int width, int height) {
//     return z * width * height + y * width + x;
// }

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
        a);
}

// CUDA kernel implementations

// 1D simulation kernel -----------------------------------------

/**
 * @brief Heat kernel for 1D simulation
 *
 * @param[in]   u0                      The heat values at the current time step (t)
 * @param[out]  u1                      The heat values at the next time step (t + dt)
 * @param[in]   boundary_condition      The boundary condition
 */
__global__ void heat_kernel_1d_sim(float *u0, float *u1, BoundaryCondition boundary_condition) {
    extern __shared__ float s_u[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int s_x = threadIdx.x + 1;

    if (x < WIDTH) {
        // Load data into shared memory
        // Central square
        s_u[s_x] = u0[x];

        // Load borders
        if (threadIdx.x == 0 && x > 0)
            s_u[0] = u0[(blockIdx.x * blockDim.x) -1];
        // Right border
        if (threadIdx.x == blockDim.x - 1 && x < WIDTH - 1)
            s_u[blockDim.x + 1] = u0[(blockIdx.x + 1) * blockDim.x];

        // Make sure all the data is loaded before computing
        __syncthreads();

        if (x > 0 && x < WIDTH - 1) {
            float u_center = s_u[s_x];
            float u_left = s_u[s_x - 1];
            float u_right = s_u[s_x + 1];

            // F-mul:
            u1[x] = u_center * (1 - 2 * RATIO);
            // F-mul-add:
            u1[x] += RATIO * u_left;
            u1[x] += RATIO * u_right;
        }
        else if (x == 0 || x == WIDTH - 1) {
            switch (boundary_condition)
            {
            case DIRICHLET:
                u1[x] = 0.0f;
                break;
            case NEUMANN:
                // Modified Neumann boundary
                if (x == 0)
                    u1[x] = s_u[s_x + 1] + HEAT_SOURCE * H2; // Left boundary
                else if (x == WIDTH - 1)
                    u1[x] = s_u[s_x - 1] + HEAT_SOURCE * H2; // Right boundary
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
 */
__global__ void heat_kernel_1d_color(float *u, uchar4 *output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < WIDTH) {
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
 * @param boundary_condition The boundary condition
 */
__global__ void heat_kernel_2d_sim(float *u0, float *u1, BoundaryCondition boundary_condition) {
    // save a blocks size worth of data into shared memory
    // + 2 for the border on left and right sides both x and y
    extern __shared__ float s_u[];

    // using blockDim.x instead of a define since if we change the block size
    // we would have to change the shared memory size 
    // (which is what we are doing in the kernel launch below)
    const int shared_bs_x = blockDim.x + 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // put into shared memory offset by 1 for the border
    int s_x = threadIdx.x + 1;
    int s_y = threadIdx.y + 1;

    if (x < WIDTH && y < HEIGHT) {
        int idx = IDX_2D(x, y, WIDTH);
        // Load data into shared memory
        // Load central square
        s_u[IDX_2D(s_x, s_y, shared_bs_x)] = u0[idx];

        // Load borders
        // Left border
        if (threadIdx.x == 0 && x > 0)
            s_u[IDX_2D(0, s_y, shared_bs_x)] = u0[IDX_2D(blockIdx.x*blockDim.x - 1, y, WIDTH)];
        // Right border
        if (threadIdx.x == blockDim.x - 1 && x < WIDTH - 1)
            s_u[IDX_2D(blockDim.x + 1, s_y, shared_bs_x)] = u0[IDX_2D((blockIdx.x + 1)*blockDim.x, y, WIDTH)];
        // Top border
        if (threadIdx.y == 0 && y > 0)
            s_u[IDX_2D(s_x, 0, shared_bs_x)] = u0[IDX_2D(x, blockIdx.y*blockDim.y - 1, WIDTH)];
        // Bottom border
        if (threadIdx.y == blockDim.y - 1 && y < HEIGHT - 1)
            s_u[IDX_2D(s_x, blockDim.y + 1, shared_bs_x)] = u0[IDX_2D(x, (blockIdx.y + 1)*blockDim.y, WIDTH)];

        // Ensure all threads have loaded their data
        __syncthreads();

        float u_center = s_u[IDX_2D(s_x, s_y, shared_bs_x)];
        float u_left = s_u[IDX_2D(s_x - 1, s_y, shared_bs_x)];
        float u_right = s_u[IDX_2D(s_x + 1, s_y, shared_bs_x)];
        float u_down = s_u[IDX_2D(s_x, s_y - 1, shared_bs_x)];
        float u_up = s_u[IDX_2D(s_x, s_y + 1, shared_bs_x)];

        if (x > 0 && x < WIDTH - 1 &&
            y > 0 && y < HEIGHT - 1) {
            u1[idx] = u_center * (1 - 4 * RATIO) +
                RATIO * (u_left + u_right + u_down + u_up);
        }
        else {
            switch (boundary_condition) {
            case DIRICHLET:
                u1[idx] = 0.0f;
                break;
            case NEUMANN:
                // Left boundary
                if (x == 0)
                    u1[idx] = u_right + HEAT_SOURCE * H2; 
                // Right boundary
                else if (x == WIDTH - 1)
                    u1[idx] = u_left + HEAT_SOURCE * H2; 
                // Top boundary
                else if (y == 0)
                    u1[idx] = u_up + HEAT_SOURCE * H2; 
                // Bottom boundary
                else if (y == HEIGHT - 1)
                    u1[idx] = u_down + HEAT_SOURCE * H2; 
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
 */
__global__ void heat_kernel_2d_color(float *u, uchar4 *output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < WIDTH && y < HEIGHT) {
        int idx = IDX_2D(x, y, WIDTH);
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
__global__ void heat_kernel_3d_sim(float* __restrict__ u0, float* __restrict__ u1, BoundaryCondition boundary_condition) {
    extern __shared__ float slice[];
    const int shared_bs_x = blockDim.x + 2;

    // Compute global indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute shared memory indices
    int s_x = threadIdx.x + 1;
    int s_y = threadIdx.y + 1;
    
    // Set up pointers to navigate through u0 and u1
    float* u0_ptr = u0 + x + y * WIDTH;
    float* u1_ptr = u1 + x + y * WIDTH;

    int stride = WIDTH * HEIGHT; // Distance between slices

    // Initialize values for z = 0
    float u_center = *u0_ptr;
    u0_ptr += stride; // Move to z = 1
    float u_infront = *u0_ptr;
    u0_ptr += stride; // Prepare for next iteration

    if (x < WIDTH && y < HEIGHT) {
        // Handle boundary at z = 0
        switch (boundary_condition) {
            case DIRICHLET:
                *u1_ptr = 0.0f;
                break;
            case NEUMANN:
                *u1_ptr = u_infront + HEAT_SOURCE * H2;
                break;
        }

        u1_ptr += stride; // Move to z = 1 in u1

        // Loop over z dimension
        float u_behind;
        for (int z = 1; z < DEPTH - 1; z++) {
            u_behind = u_center;
            u_center = u_infront;
            u_infront = *u0_ptr; // Read next slice value

            // Load u_center into shared memory
            slice[s_y * shared_bs_x + s_x] = u_center;

            // Load halo cells only if threads are at borders
            if (threadIdx.x == 0 && x > 0)
                slice[s_y * shared_bs_x + s_x - 1] = u0_ptr[-1 - stride];
            if (threadIdx.x == blockDim.x - 1 && x < WIDTH - 1)
                slice[s_y * shared_bs_x + s_x + 1] = u0_ptr[1 - stride];
            if (threadIdx.y == 0 && y > 0)
                slice[(s_y - 1) * shared_bs_x + s_x] = u0_ptr[-WIDTH - stride];
            if (threadIdx.y == blockDim.y - 1 && y < HEIGHT - 1)
                slice[(s_y + 1) * shared_bs_x + s_x] = u0_ptr[WIDTH - stride];

            __syncthreads();

            // Read neighboring values from shared memory
            float u_left = slice[s_y * shared_bs_x + s_x - 1];
            float u_right = slice[s_y * shared_bs_x + s_x + 1];
            float u_down = slice[(s_y - 1) * shared_bs_x + s_x];
            float u_up = slice[(s_y + 1) * shared_bs_x + s_x];

            // Compute new temperature
            if (x > 0 && x < (WIDTH - 1) && y > 0 && y < (HEIGHT - 1)) {
                *u1_ptr = u_center * (1 - 6 * RATIO) +
                        RATIO * (u_left + u_right + u_down + u_up + u_behind + u_infront);
            } else {
                // Handle boundaries within the computational domain
                switch (boundary_condition) {
                    case DIRICHLET:
                        *u1_ptr = 0.0f;
                        break;
                    case NEUMANN:
                        if (x == 0)
                            *u1_ptr = u_right + HEAT_SOURCE * H2;
                        else if (x == WIDTH - 1)
                            *u1_ptr = u_left + HEAT_SOURCE * H2;
                        else if (y == 0)
                            *u1_ptr = u_up + HEAT_SOURCE * H2;
                        else if (y == HEIGHT - 1)
                            *u1_ptr = u_down + HEAT_SOURCE * H2;
                        break;
                }
            }

            // Move pointers to next layer
            u0_ptr += stride;
            u1_ptr += stride;
        }

        // Handle boundary at z = DEPTH - 1
        switch (boundary_condition) {
            case DIRICHLET:
                *u1_ptr = 0.0f;
                break;
            case NEUMANN:
                *u1_ptr = u_behind + HEAT_SOURCE * H2;
                break;
        }
    }
}

/**
 * @brief Just coloring the output for 3D simulation
 *
 * @param[in]   u                       The heat values at the current time step (t)
 * @param[out]  output                  The output color values
 */
__global__ void heat_kernel_3d_color(float *u, uchar4 *output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < WIDTH && y < HEIGHT && z < DEPTH) {
        int idx = IDX_3D(x, y, z, WIDTH, HEIGHT);
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

    // Allocate texture storage
    glTexImage3D(
        GL_TEXTURE_3D,          // Target
        0,                      // Level
        GL_RGBA8,               // Internal format
        WIDTH, HEIGHT, DEPTH,   // Size
        0,                      // Border
        GL_RGBA,                // Format
        GL_UNSIGNED_BYTE,       // Type
        NULL);                  // No initial data

    glBindTexture(GL_TEXTURE_3D, 0);
}

// Render simulation
void update_sim_render()
{
    // Map PBO
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void **)&d_output, &size, cuda_pbo_resource);

    int MAX_SIM_STEPS = 10;
    if (simulation_mode == MODE_3D)
        MAX_SIM_STEPS = 2;

    for (int i=0; i<MAX_SIM_STEPS; i++){
        if (simulation_mode == MODE_1D)
        {
            // if we just did BLOCK_SIZE_X, we would only be able to do 16 threads
            // that is too few, so we need to square it
            dim3 blockSize(BLOCK_SIZE_X * BLOCK_SIZE_X * 2);
            dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x);
            int sharedMemBytes = (blockSize.x + 2) * sizeof(float);
            // simulate for all steps
            heat_kernel_1d_sim<<<gridSize, blockSize, sharedMemBytes>>>(d_u0, d_u1, boundary_condition);
            // color updated at last step
            if (i == MAX_SIM_STEPS - 1){   
                heat_kernel_1d_color<<<gridSize, blockSize>>>(d_u1, 
                    d_output);
            }
    
        }
        else if (simulation_mode == MODE_2D)
        {
            dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                        (HEIGHT + blockSize.y - 1) / blockSize.y);
            int sharedMemBytes = (blockSize.x + 2) * (blockSize.y + 2) * sizeof(float);

            heat_kernel_2d_sim<<<gridSize, blockSize, sharedMemBytes>>>(d_u0, d_u1, boundary_condition);
            
            if (i == MAX_SIM_STEPS - 1){
                heat_kernel_2d_color<<<gridSize, blockSize>>>(d_u1, 
                    d_output);
            }
        }
        else if (simulation_mode == MODE_3D)
        {

            dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                        (HEIGHT + blockSize.y - 1) / blockSize.y);
            int sharedMemBytes = (blockSize.x + 2) * (blockSize.y + 2) * sizeof(float);

            heat_kernel_3d_sim<<<gridSize, blockSize, sharedMemBytes>>>(d_u0, d_u1, boundary_condition);

            if (i == MAX_SIM_STEPS - 1){
                dim3 blockSize(BLOCK_SIZE_X/2, BLOCK_SIZE_Y/2, BLOCK_SIZE_Z/2);
                dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                            (HEIGHT + blockSize.y - 1) / blockSize.y,
                            (DEPTH + blockSize.z - 1) / blockSize.z);
                heat_kernel_3d_color<<<gridSize, blockSize>>>(d_u1, d_output);
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

    // Clear buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Bind PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    if (simulation_mode == MODE_1D)
    {
        // Set up OpenGL state for 1D rendering
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_TEXTURE_3D);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0, WIDTH, 0, 20);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        for (int i = 0; i < 10; i++)
        {
            glRasterPos2i(0, i);
            glDrawPixels(WIDTH, 1, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        }

    }
    else if (simulation_mode == MODE_2D)
    {
        // Set up OpenGL state for 2D rendering
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_TEXTURE_3D);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0, WIDTH, 0, HEIGHT);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glRasterPos2i(0, 0);
        glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    }
    else if (simulation_mode == MODE_3D)
    {
        // Set up OpenGL state for 3D rendering
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_TEXTURE_3D);

        // Bind the volume texture
        glBindTexture(GL_TEXTURE_3D, volumeTexture);

        // Transfer data from PBO to the 3D texture
        glTexSubImage3D(
            GL_TEXTURE_3D,
            0,
            0, 0, 0,
            WIDTH, HEIGHT, DEPTH,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            0);

        // Set up the projection and view matrices
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(60.0, (double) WIDTH / (double) HEIGHT, 0.01, 10000.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(
            0.0, 0.0, 2.0,
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0);

        // Render slices
        glBegin(GL_QUADS);
        for (int i = 0; i < DEPTH; ++i)
        {
            float z = -1.0f + 2.0f * i / (DEPTH - 1);
            float texZ = (float)i / (DEPTH - 1);

            glTexCoord3f(0.0f, 0.0f, texZ); glVertex3f(-1.0f, -1.0f, z);
            glTexCoord3f(1.0f, 0.0f, texZ); glVertex3f(1.0f, -1.0f, z);
            glTexCoord3f(1.0f, 1.0f, texZ); glVertex3f(1.0f, 1.0f, z);
            glTexCoord3f(0.0f, 1.0f, texZ); glVertex3f(-1.0f, 1.0f, z);
        }
        glEnd();

        // Unbind texture
        glBindTexture(GL_TEXTURE_3D, 0);
    }

    // Unbind PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Swap buffers
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

    // Normalize cursor position
    float norm_x = xpos / fb_width;
    float norm_y = ypos / fb_height;
    norm_y = 1.0f - norm_y;

    // Map to simulation grid
    int x = static_cast<int>(norm_x * WIDTH);
    int y = static_cast<int>(norm_y * HEIGHT);
    x = max(0, min(x, WIDTH - 1));
    y = max(0, min(y, HEIGHT - 1));

    if (simulation_mode == MODE_1D)
    {
        dim3 blockSize(256);
        dim3 gridSize((2 * HEAT_RADIUS + blockSize.x - 1) / blockSize.x);

        add_heat_kernel_1d<<<gridSize, blockSize>>>(d_u0, WIDTH, x);
    }
    else if (simulation_mode == MODE_2D)
    {
        dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 gridSize((2 * HEAT_RADIUS + blockSize.x - 1) / blockSize.x,
                        (2 * HEAT_RADIUS + blockSize.y - 1) / blockSize.y);

        add_heat_kernel_2d<<<gridSize, blockSize>>>(d_u0, WIDTH, HEIGHT, x, y);
    }
    else if (simulation_mode == MODE_3D)
    {
        // Calculate z (keeping your original method or adjust as shown)
        int z = static_cast<int>(((norm_x + norm_y) / 2.0f) * DEPTH);
        z = max(0, min(z, DEPTH - 1));
        
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
