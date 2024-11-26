// File: cuda_heat_equation.cu

#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <chrono>

#define WIDTH 1024
#define HEIGHT 1024
#define BLOCK_SIZE_X 32 // 16
#define BLOCK_SIZE_Y 32 // 16

#define TIME_STEP 0.05f    // Reduced time step for slower diffusion
#define DIFFUSIVITY 0.5f   // Reduced diffusivity

#define HEAT_ACCUMULATION 0.01f // Amount of heat added on mouse click

// Define a macro for the maximum of two values
#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a < b) ? a : b)

// Host variables
GLuint pbo;
struct cudaGraphicsResource* cuda_pbo_resource;
GLuint texture;
GLFWwindow* window;

// Device variables
float *d_u0, *d_u1;

// Global variables for FPS calculation
double lastTime = 0.0;
int nbFrames = 0;
double fps = 0.0;

// Function to get index in 1D array from 2D coordinates
__device__ int getIndex(int x, int y, int width) {
    return y * width + x;
}

// Evolve kernel using shared memory
__global__ void evolve_kernel(float* u0, float* u1, int width, int height, float dt, float dx2, float dy2, float a) {
    __shared__ float s_u[(BLOCK_SIZE_Y + 2) * (BLOCK_SIZE_X + 2)];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * BLOCK_SIZE_X + tx;
    int y = blockIdx.y * BLOCK_SIZE_Y + ty;

    int s_width = BLOCK_SIZE_X + 2;
    int s_x = tx + 1;
    int s_y = ty + 1;

    if (x < width && y < height) {
        // Load center point
        s_u[s_y * s_width + s_x] = u0[y * width + x];

        // Load neighbors
        if (tx == 0 && x > 0)
            s_u[s_y * s_width] = u0[y * width + x - 1];
        if (tx == BLOCK_SIZE_X - 1 && x < width -1)
            s_u[s_y * s_width + s_x + 1] = u0[y * width + x + 1];
        if (ty == 0 && y > 0)
            s_u[(s_y -1) * s_width + s_x] = u0[(y -1) * width + x];
        if (ty == BLOCK_SIZE_Y -1 && y < height -1)
            s_u[(s_y +1) * s_width + s_x] = u0[(y +1) * width + x];

        // Load corners
        if (tx == 0 && ty == 0 && x > 0 && y > 0)
            s_u[(s_y -1) * s_width] = u0[(y -1) * width + x -1];
        if (tx == BLOCK_SIZE_X -1 && ty == 0 && x < width -1 && y > 0)
            s_u[(s_y -1) * s_width + s_x +1] = u0[(y -1) * width + x +1];
        if (tx == 0 && ty == BLOCK_SIZE_Y -1 && x >0 && y < height -1)
            s_u[(s_y +1) * s_width] = u0[(y +1) * width + x -1];
        if (tx == BLOCK_SIZE_X -1 && ty == BLOCK_SIZE_Y - 1 && x < width -1 && y < height -1)
            s_u[(s_y +1) * s_width + s_x +1] = u0[(y +1) * width + x +1];

        __syncthreads();

        // Compute new value if not at boundary
        if (x > 0 && x < width -1 && y >0 && y < height -1) {
            float u_center = s_u[s_y * s_width + s_x];
            float u_left = s_u[s_y * s_width + s_x -1];
            float u_right = s_u[s_y * s_width + s_x +1];
            float u_up = s_u[(s_y -1) * s_width + s_x];
            float u_down = s_u[(s_y +1) * s_width + s_x];

            float laplacian = (u_left + u_right - 2 * u_center) / dx2 + (u_up + u_down - 2 * u_center) / dy2;

            u1[y * width + x] = u_center + a * dt * laplacian;
        } else if (x < width && y < height) {
            // Apply Dirichlet boundary condition
            u1[y * width + x] = 0.0f;
        }
    }
}

// Kernel to map heat values to colors
__global__ void heat_to_color_kernel(float* u, uchar4* output, int width, int height) {
    int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float value = u[idx];
        unsigned char color = (unsigned char)(255 * fminf(fmaxf(value, 0.0f), 0.01f)); // Clamp between 0 and 1
        
        #if 0
        output[idx] = make_uchar4(color, color, color, 255); // Grayscale
        #else
        // creating a color map
        unsigned char red = (unsigned char)(255 * value);
        unsigned char green = 0;
        unsigned char blue = (unsigned char)(255 * (1.0f - value));
        output[idx] = make_uchar4(red, green, blue, 255);
        #endif
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

    window = glfwCreateWindow(WIDTH, HEIGHT, "CUDA Heat Equation", NULL, NULL);
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

    // Create texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

// Update simulation
void update_simulation() {
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((WIDTH + BLOCK_SIZE_X -1) / BLOCK_SIZE_X, (HEIGHT + BLOCK_SIZE_Y -1) / BLOCK_SIZE_Y);

    float dx = 1.0f;
    float dy = 1.0f;
    float dx2 = dx * dx;
    float dy2 = dy * dy;
    float a = DIFFUSIVITY;

    evolve_kernel<<<gridSize, blockSize>>>(d_u0, d_u1, WIDTH, HEIGHT, TIME_STEP, dx2, dy2, a);

    // Synchronize and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in evolve_kernel: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // Swap pointers
    float* temp = d_u0;
    d_u0 = d_u1;
    d_u1 = temp;
}


// Render simulation
void render() {
    // Map PBO
    uchar4* d_output;
    size_t size;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_output, &size, cuda_pbo_resource);

    // Launch kernel to convert heat to colors
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((WIDTH + BLOCK_SIZE_X -1) / BLOCK_SIZE_X, (HEIGHT + BLOCK_SIZE_Y -1) / BLOCK_SIZE_Y);
    heat_to_color_kernel<<<gridSize, blockSize>>>(d_u0, d_output, WIDTH, HEIGHT);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error in heat_to_color_kernel: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // Unmap PBO
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

    // Draw texture
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glBegin(GL_QUADS);
    glTexCoord2f(0,0); glVertex2f(-1,-1);
    glTexCoord2f(1,0); glVertex2f(1,-1);
    glTexCoord2f(1,1); glVertex2f(1,1);
    glTexCoord2f(0,1); glVertex2f(-1,1);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glfwSwapBuffers(window);

    // FPS calculation
    double currentTime = glfwGetTime();
    nbFrames++;
    if ( currentTime - lastTime >= 1.0 ){ // If last prinf() was more than 1 sec ago
        fps = double(nbFrames) / (currentTime - lastTime);
        nbFrames = 0;
        lastTime += 1.0;

        // Update window title
        char title[256];
        sprintf(title, "CUDA Heat Equation - Width: %d Height: %d FPS: %.2f", WIDTH, HEIGHT, fps);
        glfwSetWindowTitle(window, title);
    }
}

// Function to print a small grid of heat values for debugging
void print_heat_values(int center_x, int center_y, int range) {
    size_t size = WIDTH * HEIGHT * sizeof(float);
    float* h_u = (float*)malloc(size);
    cudaMemcpy(h_u, d_u0, size, cudaMemcpyDeviceToHost);

    printf("Heat values around (%d, %d):\n", center_x, center_y);
    for (int y = center_y - range; y <= center_y + range; y++) {
        for (int x = center_x - range; x <= center_x + range; x++) {
            if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
                printf("%.2f ", h_u[y * WIDTH + x]);
            } else {
                printf("---- ");
            }
        }
        printf("\n");
    }

    free(h_u);
}

// Mouse callback
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        // Get mouse position
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        // Convert to grid coordinates
        int fb_width, fb_height;
        glfwGetFramebufferSize(window, &fb_width, &fb_height);
        int x = (int)(xpos * WIDTH / fb_width);
        int y = (int)(ypos * HEIGHT / fb_height);

        // Debug print
        printf("Mouse clicked at (%d, %d)\n", x, y);

        size_t size = WIDTH * HEIGHT * sizeof(float);
        float* h_u = (float*)malloc(size);

        // Copy current heat values to host
        cudaMemcpy(h_u, d_u0, size, cudaMemcpyDeviceToHost);

        // Add heat at the mouse position
        const int radius = 5;
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
            int xx = x + dx;
            int yy = y + dy;
            if (xx >= 0 && 
                xx < WIDTH && 
                yy >= 0 && 
                yy < HEIGHT && 
                dx * dx + dy * dy <= radius * radius) {
                // Accumulate heat and clamp to 1.0f
                h_u[yy * WIDTH + xx] = MIN(h_u[yy * WIDTH + xx] + HEAT_ACCUMULATION, 1.0f);
            }
            }
        }

        // Debug: Print the updated heat value at the center
        printf("Updated heat at (%d, %d): %f\n", x, y, h_u[y * WIDTH + x]);
        // After copying back to device, print heat values around the clicked point
        print_heat_values(x, y, 2);

        // Copy updated heat values back to device
        cudaMemcpy(d_u0, h_u, size, cudaMemcpyHostToDevice);
        free(h_u);

    }
}

int main() {
    init_opengl();
    init_simulation();
    
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Initialize timing
    lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        update_simulation();
        render();
    }

    // Cleanup
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture);
    cudaFree(d_u0);
    cudaFree(d_u1);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}