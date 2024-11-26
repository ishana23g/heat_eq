// File: cuda_heat_equation_1d.cu

#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define WIDTH 1024
#define TIME_STEP 0.1f
#define DIFFUSIVITY 0.5f
#define HEAT_SOURCE 1.0f

// Host variables
GLuint pbo;
struct cudaGraphicsResource* cuda_pbo_resource;
GLFWwindow* window;

// Device variables
float *d_u0, *d_u1;
uchar4 *d_output;

// CUDA kernel to compute the heat equation
__global__ void heat_kernel(float* u0, float* u1, int width, float dt, float dx2, float a) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x > 0 && x < width - 1) {
        float u_center = u0[x];
        float u_left = u0[x - 1];
        float u_right = u0[x + 1];

        float laplacian = (u_left + u_right - 2 * u_center) / dx2;
        u1[x] = u_center + a * dt * laplacian;
    } else {
        // Boundary conditions
        u1[x] = 0.0f;
    }
}

// CUDA kernel to map heat values to colors
__global__ void heat_to_color_kernel(float* u, uchar4* output, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < width) {
        float value = u[x];
        unsigned char color = (unsigned char)(255 * fminf(fmaxf(value, 0.0f), 1.0f));

        output[x] = make_uchar4(color, 0, 255 - color, 255);
    }
}

// Initialize the simulation
void init_simulation() {
    size_t size = WIDTH * sizeof(float);
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

    window = glfwCreateWindow(WIDTH, 100, "1D Heat Equation", NULL, NULL);
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
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Register PBO with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

// Update simulation
void update_simulation() {
    dim3 blockSize(256);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x);

    float dx = 1.0f;
    float dx2 = dx * dx;
    float a = DIFFUSIVITY;

    heat_kernel<<<gridSize, blockSize>>>(d_u0, d_u1, WIDTH, TIME_STEP, dx2, a);
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
    dim3 blockSize(256);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x);
    heat_to_color_kernel<<<gridSize, blockSize>>>(d_u0, d_output, WIDTH);
    cudaDeviceSynchronize();

    // Unmap PBO
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

    // Draw pixels
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    for (int i = 0; i < 10; ++i) {
        glDrawPixels(WIDTH, 1, GL_RGBA, GL_UNSIGNED_BYTE, (const void*)(i * WIDTH * sizeof(uchar4)));
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glfwSwapBuffers(window);
}

// Mouse callback
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        // Get mouse position
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        // Convert to grid coordinate
        int fb_width, fb_height;
        glfwGetFramebufferSize(window, &fb_width, &fb_height);
        int x = (int)(xpos * WIDTH / fb_width);

        // Add heat source
        size_t size = WIDTH * sizeof(float);
        float* h_u = (float*)malloc(size);

        cudaMemcpy(h_u, d_u0, size, cudaMemcpyDeviceToHost);

        const int radius = 5;
        for (int dx = -radius; dx <= radius; dx++) {
            int idx = x + dx;
            if (idx >= 0 && idx < WIDTH) {
                h_u[idx] += HEAT_SOURCE;
            }
        }

        cudaMemcpy(d_u0, h_u, size, cudaMemcpyHostToDevice);
        free(h_u);
    }
}

int main() {
    init_opengl();
    init_simulation();

    glfwSetMouseButtonCallback(window, mouse_button_callback);

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