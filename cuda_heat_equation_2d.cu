// File: cuda_heat_equation_2d_drag.cu

#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define WIDTH 512
#define HEIGHT 512
#define TIME_STEP 0.1f
#define DIFFUSIVITY 0.1f
#define HEAT_SOURCE 5.0f

// Host variables
GLuint pbo;
struct cudaGraphicsResource* cuda_pbo_resource;
GLFWwindow* window;

// Device variables
float *d_u0, *d_u1;
uchar4 *d_output;

// Variables to track mouse state
bool is_mouse_pressed = false;

// CUDA kernel to compute the heat equation
__global__ void heat_kernel(float* u0, float* u1, int width, int height, float dt, float dx2, float dy2, float a) {
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
        // Boundary conditions
        int idx = y * width + x;
        u1[idx] = 0.0f;
    }
}

// CUDA kernel to map heat values to colors
__global__ void heat_to_color_kernel(float* u, uchar4* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float value = u[idx];
        unsigned char color = (unsigned char)(255 * fminf(fmaxf(value / HEAT_SOURCE, 0.0f), 1.0f));

        output[idx] = make_uchar4(color, 0, 255 - color, 255);
    }
}

// CUDA kernel to add heat at a specific location
__global__ void add_heat_kernel(float* u, int width, int height, int cx, int cy, int radius, float heat) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x - radius;
    int ty = blockIdx.y * blockDim.y + threadIdx.y - radius;

    int x = cx + tx;
    int y = cy + ty;

    if (x >= 0 && x < width && y >= 0 && y < height) {
        if (tx * tx + ty * ty <= radius * radius) {
            int idx = y * width + x;
            atomicAdd(&u[idx], heat);
        }
    }
}

// Initialize the simulation
void init_simulation() {
    size_t size = WIDTH * HEIGHT * sizeof(float);
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

    window = glfwCreateWindow(WIDTH, HEIGHT, "2D Heat Equation - Click and Drag", NULL, NULL);
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
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Register PBO with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

// Update simulation
void update_simulation() {
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);

    float dx = 1.0f;
    float dy = 1.0f;
    float dx2 = dx * dx;
    float dy2 = dy * dy;
    float a = DIFFUSIVITY;

    heat_kernel<<<gridSize, blockSize>>>(d_u0, d_u1, WIDTH, HEIGHT, TIME_STEP, dx2, dy2, a);
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

    // Launch kernel to convert heat to colors
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);

    heat_to_color_kernel<<<gridSize, blockSize>>>(d_u0, d_output, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    // Unmap PBO
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

    // Draw pixels
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    glRasterPos2i(-1, -1);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glfwSwapBuffers(window);
}

// Cursor position callback - handle mouse movement
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (is_mouse_pressed) {
        // Convert to grid coordinates
        int fb_width, fb_height;
        glfwGetFramebufferSize(window, &fb_width, &fb_height);
        int x = (int)(xpos * WIDTH / fb_width);
        int y = (int)((fb_height - ypos) * HEIGHT / fb_height); // Invert y-axis

        // Add heat source on the device
        const int radius = 5;
        dim3 blockSize(16, 16);
        dim3 gridSize((2 * radius + blockSize.x - 1) / blockSize.x,
                      (2 * radius + blockSize.y - 1) / blockSize.y);

        // Launch kernel to add heat at the current mouse position
        add_heat_kernel<<<gridSize, blockSize>>>(d_u0, WIDTH, HEIGHT, x, y, radius, HEAT_SOURCE);
        cudaDeviceSynchronize();
    }
}

// Mouse button callback - handle mouse press and release
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            is_mouse_pressed = true;

            // Add heat immediately at the point of click
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);

            // Convert to grid coordinates
            int fb_width, fb_height;
            glfwGetFramebufferSize(window, &fb_width, &fb_height);
            int x = (int)(xpos * WIDTH / fb_width);
            int y = (int)((fb_height - ypos) * HEIGHT / fb_height); // Invert y-axis

            // Add heat source on the device
            const int radius = 5;
            dim3 blockSize(16, 16);
            dim3 gridSize((2 * radius + blockSize.x - 1) / blockSize.x,
                          (2 * radius + blockSize.y - 1) / blockSize.y);

            // Launch kernel to add heat at the clicked position
            add_heat_kernel<<<gridSize, blockSize>>>(d_u0, WIDTH, HEIGHT, x, y, radius, HEAT_SOURCE);
            cudaDeviceSynchronize();
        } else if (action == GLFW_RELEASE) {
            is_mouse_pressed = false;
        }
    }
}

int main() {
    init_opengl();
    init_simulation();

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