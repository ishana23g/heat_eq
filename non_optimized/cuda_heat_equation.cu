// File: cuda_heat_equation_generalized.cu

#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Simulation settings
#define WIDTH 1000
#define HEIGHT 1000
#define TIME_STEP 0.25f
#define DIFFUSIVITY 1.0f
#define HEAT_SOURCE 5.0f
#define DX 1.0f
#define DY 1.0f
#define HEAT_RADIUS 5

// CUDA block size
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

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
    MODE_2D
};
SimulationMode simulation_mode = MODE_2D;

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
void update_simulation();
void render();
void reset_simulation();
void keyboard_callback(GLFWwindow *window, int key, int scancode, int action, int mods);
void cursor_position_callback(GLFWwindow *window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);

// CUDA kernels
__global__ void heat_kernel_1d(float *u0, float *u1,
                               int width, float dt,
                               float dx2, float a,
                               BoundaryCondition boundary_condition);
__global__ void heat_to_color_kernel_1d(float *u, uchar4 *output, int width);
__global__ void add_heat_kernel_1d(float *u, int width, int x);
__global__ void heat_kernel_2d(float *u0, float *u1,
                               int width, int height,
                               float dt, float dx2,
                               float dy2, float a,
                               BoundaryCondition boundary_condition);
__global__ void heat_to_color_kernel_2d(float *u, uchar4 *output, int width, int height);
__global__ void add_heat_kernel_2d(float *u, int width, int height, int cx, int cy);

// Color Functions

// CUDA kernel implementations
/**
 * @brief Convert 2D index layout to unrolled 1D layout
 *
 * @param[in] x      Row index
 * @param[in] y      Column index
 * @param[in] width  The width of the area
 *
 * @returns An index in the unrolled 1D array.
 */
int __host__ __device__ getIndex(const int x, const int y, const int width)
{
    return y * width + x;
}

// Heat kernel for 1D simulation
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
__global__ void heat_kernel_1d(float *u0, float *u1, int width, float dt, float dx2, float a, BoundaryCondition boundary_condition)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x > 0 && x < width - 1)
    {
        float u_center = u0[x];
        float u_left = u0[x - 1];
        float u_right = u0[x + 1];

        float laplacian = (u_left + u_right - 2 * u_center) / dx2;

        u1[x] = u_center + a * dt * laplacian;
    }
    else if (x < width)
    {
        switch (boundary_condition)
        {
        case DIRICHLET:
            u1[x] = 0.0f;
            break;
        case NEUMANN:
            // Modified Neumann boundary
            if (x == 0)
            {
                u1[x] = u1[x + 1] + HEAT_SOURCE * (dx2); // Left boundary
            }
            else if (x == width - 1)
            {
                u1[x] = u1[x - 1] + HEAT_SOURCE * (dx2); // Right boundary
            }
            break;
        }
    }
}

// Heat to color kernel for 1D simulation
__global__ void heat_to_color_kernel_1d(float *u, uchar4 *output, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < width)
    {
        float value = u[x];
        unsigned char color = (unsigned char)(255 * fminf(fmaxf(value / HEAT_SOURCE, 0.0f), 1.0f));

        output[x] = make_uchar4(color, 0, 255 - color, 255);
    }
}

// Heat kernel for 2D simulation
__global__ void heat_kernel_2d(float *u0, float *u1, int width, int height, float dt, float dx2, float dy2, float a, BoundaryCondition boundary_condition)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
    {
        int idx = y * width + x;
        int idx_left = idx - 1;
        int idx_right = idx + 1;
        int idx_up = idx - width;
        int idx_down = idx + width;

        float u_center = u0[idx];
        float u_left = u0[idx_left];
        float u_right = u0[idx_right];
        float u_up = u0[idx_up];
        float u_down = u0[idx_down];

        float laplacian = (u_left + u_right - 2 * u_center) / dx2 +
                          (u_up + u_down - 2 * u_center) / dy2;

        u1[idx] = u_center + a * dt * laplacian;
    }
    else if (x < width && y < height)
    {
        int idx = getIndex(x, y, width);

        switch (boundary_condition)
        {
        case DIRICHLET:
            u1[idx] = 0.0f;
            break;
        case NEUMANN:
            if (x == 0)
            {
                u1[idx] = u1[getIndex(x + 1, y, width)] + HEAT_SOURCE * dx2; // Left boundary
            }
            else if (x == width - 1)
            {
                u1[idx] = u1[getIndex(x - 1, y, width)] + HEAT_SOURCE * dx2; // Right boundary
            }
            else if (y == 0)
            {
                u1[idx] = u1[getIndex(x, y + 1, width)] + HEAT_SOURCE * dy2; // Top boundary
            }
            else if (y == height - 1)
            {
                u1[idx] = u1[getIndex(x, y - 1, width)] + HEAT_SOURCE * dy2; // Bottom boundary
            }
            break;
        }
    }
}

// Heat to color kernel for 2D simulation
__global__ void heat_to_color_kernel_2d(float *u, uchar4 *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        float value = u[idx];
        unsigned char color = (unsigned char)(255 * fminf(fmaxf(value / HEAT_SOURCE, 0.0f), 1.0f));

        output[idx] = make_uchar4(color, 0, 255 - color, 255);
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
            atomicAdd(&u[idx], HEAT_SOURCE);
        }
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
            atomicAdd(&u[idx], HEAT_SOURCE);
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
    else
    {
        size = WIDTH * HEIGHT * sizeof(float);
    }

    cudaMalloc((void **)&d_u0, size);
    cudaMalloc((void **)&d_u1, size);

    cudaMemset(d_u0, 0, size);
    cudaMemset(d_u1, 0, size);
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
    size_t pbo_size = (simulation_mode == MODE_1D) ? WIDTH * sizeof(uchar4) : WIDTH * HEIGHT * sizeof(uchar4);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, pbo_size, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Register PBO with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

// Update simulation
void update_simulation()
{
    if (simulation_mode == MODE_1D)
    {
        dim3 blockSize(BLOCK_SIZE_X*BLOCK_SIZE_X);
        dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x);

        heat_kernel_1d<<<gridSize, blockSize>>>(
            d_u0, d_u1, WIDTH, 
            TIME_STEP, DX * DX, 
            DIFFUSIVITY, boundary_condition);
    }
    else
    {
        dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                      (HEIGHT + blockSize.y - 1) / blockSize.y);

        heat_kernel_2d<<<gridSize, blockSize>>>(
            d_u0, d_u1, WIDTH, HEIGHT, 
            TIME_STEP, DX * DX, DY * DY, 
            DIFFUSIVITY, boundary_condition);
    }

    cudaDeviceSynchronize();

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
}

// Render simulation
void render()
{
    // Map PBO
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void **)&d_output, &size, cuda_pbo_resource);

    if (simulation_mode == MODE_1D)
    {
        dim3 blockSize(BLOCK_SIZE_X);
        dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x);

        heat_to_color_kernel_1d<<<gridSize, blockSize>>>(d_u0, d_output, WIDTH);
    }
    else
    {
        dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                      (HEIGHT + blockSize.y - 1) / blockSize.y);

        heat_to_color_kernel_2d<<<gridSize, blockSize>>>(d_u0, d_output, WIDTH, HEIGHT);
    }

    cudaDeviceSynchronize();

    // Unmap PBO
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

    // Draw pixels
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    if (simulation_mode == MODE_1D)
    {   
        for (int i = 0; i < 20; i++)
        {
            glRasterPos2f(-1.0f, -1.0f + i * 0.02f);
            glDrawPixels(WIDTH, 1, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        }
    }
    else
    {
        glRasterPos2i(-1, -1);
        glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
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
    else
    {
        size = WIDTH * HEIGHT * sizeof(float);
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
        case GLFW_KEY_B:
            boundary_condition = (boundary_condition == DIRICHLET) ? NEUMANN : DIRICHLET;
            printf("Switched boundary condition to %s\n",
                   (boundary_condition == DIRICHLET) ? "Dirichlet" : "Neumann");
            break;
        case GLFW_KEY_R:
            reset_simulation();
            printf("Simulation reset\n");
            break;
        default:
            break;
        }
    }
}


void add_heat_launcher(int x, int y)
{
    if (simulation_mode == MODE_1D)
    {
        dim3 blockSize(BLOCK_SIZE_X*BLOCK_SIZE_X);
        dim3 gridSize((2 * HEAT_RADIUS + blockSize.x - 1) / blockSize.x);

        add_heat_kernel_1d<<<gridSize, blockSize>>>(d_u0, WIDTH, x);
    }
    else
    {
        dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 gridSize((2 * HEAT_RADIUS + blockSize.x - 1) / blockSize.x,
                      (2 * HEAT_RADIUS + blockSize.y - 1) / blockSize.y);

        add_heat_kernel_2d<<<gridSize, blockSize>>>(d_u0, WIDTH, HEIGHT, x, y);
    }

    cudaDeviceSynchronize();
}

// Cursor position callback
void cursor_position_callback(GLFWwindow *window, double xpos, double ypos)
{
    if (is_mouse_pressed)
    {
        int fb_width, fb_height;
        glfwGetFramebufferSize(window, &fb_width, &fb_height);
        int x = (int)(xpos * WIDTH / fb_width);
        int y = (int)((fb_height - ypos) * HEIGHT / fb_height); // Invert y-axis

        add_heat_launcher(x, y);
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

// Main function
int main(int argc, char **argv)
{
    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)
        {
            // switch modes, 1d or 2d
            if (strcmp(argv[i + 1], "1d") == 0)
            {
                simulation_mode = MODE_1D;
            }
            else if (strcmp(argv[i + 1], "2d") == 0)
            {
                simulation_mode = MODE_2D;
            }
            ++i;
        }
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc)
        {
            // switch boundary conditions, dirichlet or neumann
            if (strcmp(argv[i + 1], "d") == 0)
            {
                boundary_condition = DIRICHLET;
            }
            else if (strcmp(argv[i + 1], "n") == 0)
            {
                boundary_condition = NEUMANN;
            }
            ++i;
        }
        else if (strcmp(argv[i], "-d") == 0)
        {
            // debug mode. Sets a hard coded value for seeing how long the simulation takes to run
            debug_mode = true;
            // the next two numbers are time, and percent chance to add heat
            if (i + 2 < argc)
            {
                MAX_TIME_STEPS = atoi(argv[i + 1]);
                PERCENT_ADD_HEAT_CHANCE = atoi(argv[i + 2]);
                if (MAX_TIME_STEPS < 0)
                {
                    MAX_TIME_STEPS = 100;
                }
                if (PERCENT_ADD_HEAT_CHANCE < 0 || PERCENT_ADD_HEAT_CHANCE > 100)
                {
                    PERCENT_ADD_HEAT_CHANCE = 40;
                }
                i += 2;
            }
        }
    }

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
            update_simulation();
            render();

            // randomlly add heat to the simulation, not at all time steps
            if (rand() % 100 < PERCENT_ADD_HEAT_CHANCE)
            {
                int x = rand() % WIDTH;
                int y = rand() % HEIGHT;
                dim3 blockSize(256);
                dim3 gridSize((2 * HEAT_RADIUS + blockSize.x - 1) / blockSize.x,
                              (2 * HEAT_RADIUS + blockSize.y - 1) / blockSize.y);

                add_heat_kernel_2d<<<gridSize, blockSize>>>(d_u0, WIDTH, HEIGHT, x, y);
            }
        }
    }
    else
    {
        // Main loop
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            update_simulation();
            render();
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