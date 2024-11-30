#include <GL/glut.h>
#define GL_GLEXT_PROTOTYPES

#include <GL/gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>

// Grid size
const int GRID_SIZE = 64;
const float TIME_STEP = 0.01f;

// CUDA variables
float *d_current, *d_next;

// OpenGL variables
GLuint vbo;
cudaGraphicsResource_t cuda_vbo_resource;

// CUDA Kernel for 3D Heat Equation
__global__ void heatEquation3D(float *current, float *next, int grid_size, float time_step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && x < grid_size - 1 && 
        y > 0 && y < grid_size - 1 && 
        z > 0 && z < grid_size - 1) {

        int idx = x + y * grid_size + z * grid_size * grid_size;

        float up = current[idx - grid_size];
        float down = current[idx + grid_size];
        float left = current[idx - 1];
        float right = current[idx + 1];
        float front = current[idx - grid_size * grid_size];
        float back = current[idx + grid_size * grid_size];

        next[idx] = current[idx] + time_step * (up + down + left + right + front + back - 6 * current[idx]);
    }
}

// Initialize Grid
void initializeGrid(float *grid, int grid_size) {
    for (int z = 0; z < grid_size; z++) {
        for (int y = 0; y < grid_size; y++) {
            for (int x = 0; x < grid_size; x++) {
                int idx = x + y * grid_size + z * grid_size * grid_size;
                grid[idx] = (x == grid_size / 2 && y == grid_size / 2 && z == grid_size / 2) ? 1.0f : 0.0f;
            }
        }
    }
}

// OpenGL Rendering
void renderScene() {
    // Map OpenGL VBO to CUDA
    float *vbo_ptr;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&vbo_ptr, nullptr, cuda_vbo_resource);

    // Copy data from CUDA grid to OpenGL VBO
    cudaMemcpy(vbo_ptr, d_current, GRID_SIZE * GRID_SIZE * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);

    // Unmap OpenGL VBO
    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

    // Render points
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPointSize(2.0f);
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glDrawArrays(GL_POINTS, 0, GRID_SIZE * GRID_SIZE * GRID_SIZE);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
}

// Timer Function for CUDA Execution
void timerFunction(int value) {
    dim3 block(8, 8, 8);
    dim3 grid((GRID_SIZE + block.x - 1) / block.x,
              (GRID_SIZE + block.y - 1) / block.y,
              (GRID_SIZE + block.z - 1) / block.z);

    // Launch the CUDA kernel
    heatEquation3D<<<grid, block>>>(d_current, d_next, GRID_SIZE, TIME_STEP);

    // Swap buffers
    std::swap(d_current, d_next);

    glutPostRedisplay();
    glutTimerFunc(16, timerFunction, 0);
}

// OpenGL Initialization
void initOpenGL() {
    glEnable(GL_DEPTH_TEST);

    // Generate OpenGL VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, GRID_SIZE * GRID_SIZE * GRID_SIZE * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

// Main Function
int main(int argc, char **argv) {
    // Allocate memory for grids on the GPU
    cudaMalloc(&d_current, GRID_SIZE * GRID_SIZE * GRID_SIZE * sizeof(float));
    cudaMalloc(&d_next, GRID_SIZE * GRID_SIZE * GRID_SIZE * sizeof(float));

    float *h_grid = new float[GRID_SIZE * GRID_SIZE * GRID_SIZE];
    initializeGrid(h_grid, GRID_SIZE);
    cudaMemcpy(d_current, h_grid, GRID_SIZE * GRID_SIZE * GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // OpenGL initialization
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 800);
    glutCreateWindow("3D Heat Equation with CUDA");

    initOpenGL();

    // Set rendering and timer functions
    glutDisplayFunc(renderScene);
    glutTimerFunc(16, timerFunction, 0);

    glutMainLoop();

    // Cleanup
    delete[] h_grid;
    cudaFree(d_current);
    cudaFree(d_next);
    cudaGraphicsUnregisterResource(cuda_vbo_resource);
    glDeleteBuffers(1, &vbo);

    return 0;
}
