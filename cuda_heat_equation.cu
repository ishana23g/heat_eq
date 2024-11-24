#include <stdio.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define WIDTH 512
#define HEIGHT 512
#define TIME_STEP 0.1f

// OpenGL resources
GLuint pbo;                // Pixel Buffer Object
GLuint texture;            // Texture to display
struct cudaGraphicsResource* cudaPboResource;

// CUDA kernel for heat equation
__global__ void heatKernel(float* grid, float* newGrid, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;

        // Heat equation: simple 5-point stencil
        float center = grid[idx];
        float left = grid[idx - 1];
        float right = grid[idx + 1];
        float top = grid[idx - width];
        float bottom = grid[idx + width];

        newGrid[idx] = center + TIME_STEP * (left + right + top + bottom - 4.0f * center);
    }
}

// Utility to initialize the grid with a hot spot
void initializeGrid(float* grid, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if ((x - width / 2) * (x - width / 2) + (y - height / 2) * (y - height / 2) < 1000) {
                grid[y * width + x] = 1.0f;  // Hot spot in the center
            } else {
                grid[y * width + x] = 0.0f;  // Rest is cold
            }
        }
    }
}

// CUDA function to copy simulation data to OpenGL texture
__global__ void copyToTexture(uchar4* texture, float* grid, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float value = grid[idx];
        unsigned char color = (unsigned char)(255 * value);

        texture[idx] = make_uchar4(color, color, color, 255);
    }
}

// OpenGL setup
void initOpenGL(GLFWwindow** window) {
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    *window = glfwCreateWindow(WIDTH, HEIGHT, "Heat Equation Simulation", NULL, NULL);
    if (!*window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(*window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        exit(EXIT_FAILURE);
    }

    glViewport(0, 0, WIDTH, HEIGHT);
    glDisable(GL_DEPTH_TEST);
}

// Create OpenGL texture and PBO
void createTextureAndPBO() {
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

// Render OpenGL texture
void renderTexture() {
    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();
    glfwSwapBuffers(glfwGetCurrentContext());
}

int main() {
    // Initialize OpenGL
    GLFWwindow* window;
    initOpenGL(&window);

    // Create OpenGL texture and PBO
    createTextureAndPBO();

    // Allocate CUDA memory for the grid
    float* d_grid, * d_newGrid;
    size_t gridSize = WIDTH * HEIGHT * sizeof(float);
    cudaMalloc(&d_grid, gridSize);
    cudaMalloc(&d_newGrid, gridSize);

    // Initialize grid
    float* h_grid = (float*)malloc(gridSize);
    initializeGrid(h_grid, WIDTH, HEIGHT);
    cudaMemcpy(d_grid, h_grid, gridSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSizeDim((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    while (!glfwWindowShouldClose(window)) {
        // Launch CUDA kernel to compute heat diffusion
        heatKernel<<<gridSizeDim, blockSize>>>(d_grid, d_newGrid, WIDTH, HEIGHT);

        // Swap grids
        float* temp = d_grid;
        d_grid = d_newGrid;
        d_newGrid = temp;

        // Map PBO to CUDA and copy simulation data to texture
        cudaGraphicsMapResources(1, &cudaPboResource, 0);
        uchar4* d_texture;
        size_t numBytes;
        cudaGraphicsResourceGetMappedPointer((void**)&d_texture, &numBytes, cudaPboResource);
        copyToTexture<<<gridSizeDim, blockSize>>>(d_texture, d_grid, WIDTH, HEIGHT);
        cudaGraphicsUnmapResources(1, &cudaPboResource, 0);

        // Render OpenGL texture
        renderTexture();

        // Poll events
        glfwPollEvents();
    }

    // Clean up
    cudaGraphicsUnregisterResource(cudaPboResource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture);
    cudaFree(d_grid);
    cudaFree(d_newGrid);
    free(h_grid);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
