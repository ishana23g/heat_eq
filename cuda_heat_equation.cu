// File: cuda_heat_equation_generalized.cu

#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Simulation settings
#define WIDTH 512
#define HEIGHT 512
#define TIME_STEP 0.1f
#define DIFFUSIVITY 0.1f
#define HEAT_SOURCE 5.0f

#define HEAT_RADIUS 5

// CUDA block size
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Host variables
GLuint pbo;
struct cudaGraphicsResource* cuda_pbo_resource;
GLFWwindow* window;

// Device variables
float *d_u0, *d_u1;
uchar4 *d_output;

// Mouse state
bool is_mouse_pressed = false;

// Simulation modes
enum SimulationMode { MODE_1D, MODE_2D };
SimulationMode simulation_mode = MODE_2D;

// Boundary conditions
enum BoundaryCondition { DIRICHLET, NEUMANN };
BoundaryCondition boundary_condition = DIRICHLET;

// Function prototypes
void init_simulation();
void init_opengl();
void update_simulation();
void render();
void reset_simulation();
void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

// CUDA kernels
__global__ void heat_kernel_1d(float* u0, float* u1, int width, float dt, float dx2, float a, BoundaryCondition boundary_condition);
__global__ void heat_to_color_kernel_1d(float* u, uchar4* output, int width);
__global__ void heat_kernel_2d(float* u0, float* u1, int width, int height, float dt, float dx2, float dy2, float a, BoundaryCondition boundary_condition);
__global__ void heat_to_color_kernel_2d(float* u, uchar4* output, int width, int height);
__global__ void add_heat_kernel_1d(float* u, int width, int x);
__global__ void add_heat_kernel_2d(float* u, int width, int height, int cx, int cy);

// Initialize the simulation
void init_simulation() {
    size_t size;
    if (simulation_mode == MODE_1D) {
        size = WIDTH * sizeof(float);
    } else {
        size = WIDTH * HEIGHT * sizeof(float);
    }

    cudaMalloc((void**)&d_u0, size);
    cudaMalloc((void**)&d_u1, size);

    cudaMemset(d_u0, 0, size);
    cudaMemset(d_u1, 0, size);
}

// Initialize OpenGL
void init_opengl() {
    if (!glfwInit()) {
        printf("Failed to initialize GLFW\n");
        exit(-1);
    }

    int window_width = WIDTH;
    int window_height = (simulation_mode == MODE_1D) ? 100 : HEIGHT;
    window = glfwCreateWindow(window_width, window_height, "CUDA Heat Equation", NULL, NULL);
    if (!window) {
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
void update_simulation() {
    float dx = 1.0f;
    float dy = 1.0f;
    float dx2 = dx * dx;
    float dy2 = dy * dy;
    float a = DIFFUSIVITY;

    if (simulation_mode == MODE_1D) {
        dim3 blockSize(256);
        dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x);

        heat_kernel_1d<<<gridSize, blockSize>>>(d_u0, d_u1, WIDTH, TIME_STEP, dx2, a, boundary_condition);
    } else {
        dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                      (HEIGHT + blockSize.y - 1) / blockSize.y);

        heat_kernel_2d<<<gridSize, blockSize>>>(d_u0, d_u1, WIDTH, HEIGHT, TIME_STEP, dx2, dy2, a, boundary_condition);
    }

    cudaDeviceSynchronize();

    // Swap pointers
    float* temp = d_u0;
    d_u0 = d_u1;
    d_u1 = temp;
}

// Render simulation
void render() {
    // Map PBO
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void**)&d_output, &size, cuda_pbo_resource);

    if (simulation_mode == MODE_1D) {
        dim3 blockSize(256);
        dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x);

        heat_to_color_kernel_1d<<<gridSize, blockSize>>>(d_u0, d_output, WIDTH);
    } else {
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
    if (simulation_mode == MODE_1D) {
        glDrawPixels(WIDTH, 1, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    } else {
        glRasterPos2i(-1, -1);
        glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glfwSwapBuffers(window);
}

// Reset simulation
void reset_simulation() {
    size_t size;
    if (simulation_mode == MODE_1D) {
        size = WIDTH * sizeof(float);
    } else {
        size = WIDTH * HEIGHT * sizeof(float);
    }
    cudaMemset(d_u0, 0, size);
    cudaMemset(d_u1, 0, size);
}

// Keyboard callback
void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        switch (key) {
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

// Cursor position callback
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (is_mouse_pressed) {
        int fb_width, fb_height;
        glfwGetFramebufferSize(window, &fb_width, &fb_height);

        if (simulation_mode == MODE_1D) {
            int x = (int)(xpos * WIDTH / fb_width);
            
            dim3 blockSize(256);
            dim3 gridSize((2 * HEAT_RADIUS + blockSize.x - 1) / blockSize.x);

            add_heat_kernel_1d<<<gridSize, blockSize>>>(d_u0, WIDTH, x);
        } else {
            int x = (int)(xpos * WIDTH / fb_width);
            int y = (int)((fb_height - ypos) * HEIGHT / fb_height); // Invert y-axis
            dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            dim3 gridSize((2 * HEAT_RADIUS + blockSize.x - 1) / blockSize.x,
                          (2 * HEAT_RADIUS + blockSize.y - 1) / blockSize.y);

            add_heat_kernel_2d<<<gridSize, blockSize>>>(d_u0, WIDTH, HEIGHT, x, y);
        }

        cudaDeviceSynchronize();
    }
}

// Mouse button callback
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            is_mouse_pressed = true;
            cursor_position_callback(window, 0, 0); // Trigger heat addition
        } else if (action == GLFW_RELEASE) {
            is_mouse_pressed = false;
        }
    }
}

// Main function
int main() {
    init_opengl();
    init_simulation();

    glfwSetKeyCallback(window, keyboard_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        update_simulation();
        render();
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

// CUDA kernel implementations

// Heat kernel for 1D simulation
__global__ void heat_kernel_1d(float* u0, float* u1, int width, float dt, float dx2, float a, BoundaryCondition boundary_condition) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x > 0 && x < width - 1) {
        float u_center = u0[x];
        float u_left = u0[x - 1];
        float u_right = u0[x + 1];

        float laplacian = (u_left + u_right - 2 * u_center) / dx2;

        u1[x] = u_center + a * dt * laplacian;
    } else if (x == 0 || x == width - 1) {
        // Boundary conditions
        if (boundary_condition == DIRICHLET) {
            u0[x] = 0.0f;
        } else if (boundary_condition == NEUMANN) {
            u1[x] = HEAT_SOURCE;
        }
    }
}

// Heat to color kernel for 1D simulation
__global__ void heat_to_color_kernel_1d(float* u, uchar4* output, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < width) {
        float value = u[x];
        unsigned char color = (unsigned char)(255 * fminf(fmaxf(value / HEAT_SOURCE, 0.0f), 1.0f));

        output[x] = make_uchar4(color, 0, 255 - color, 255);
    }
}

// Heat kernel for 2D simulation
__global__ void heat_kernel_2d(float* u0, float* u1, int width, int height, float dt, float dx2, float dy2, float a, BoundaryCondition boundary_condition) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
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
    } else if (x < width && y < height) {
        int idx = y * width + x;
        if (boundary_condition == DIRICHLET) {
            u0[idx] = 0.0f;
        } else if (boundary_condition == NEUMANN) {
            u1[idx] = HEAT_SOURCE;
        }
    }
}

// Heat to color kernel for 2D simulation
__global__ void heat_to_color_kernel_2d(float* u, uchar4* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float value = u[idx];
        unsigned char color = (unsigned char)(255 * fminf(fmaxf(value / HEAT_SOURCE, 0.0f), 1.0f));

        output[idx] = make_uchar4(color, 0, 255 - color, 255);
    }
}

// Add heat kernel for 1D simulation
__global__ void add_heat_kernel_1d(float* u, int width, int x) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x - HEAT_RADIUS;
    int idx = x + tx;

    if (idx >= 0 && idx < width) {
        if (abs(tx) <= HEAT_RADIUS) {
            atomicAdd(&u[idx], HEAT_SOURCE);
        }
    }
}

// Add heat kernel for 2D simulation
__global__ void add_heat_kernel_2d(float* u, int width, int height, int cx, int cy) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x - HEAT_RADIUS;
    int ty = blockIdx.y * blockDim.y + threadIdx.y - HEAT_RADIUS;

    int x = cx + tx;
    int y = cy + ty;

    if (x >= 0 && x < width && y >= 0 && y < height) {
        if (tx * tx + ty * ty <= HEAT_RADIUS * HEAT_RADIUS) {
            int idx = y * width + x;
            atomicAdd(&u[idx], HEAT_SOURCE);
        }
    }
}