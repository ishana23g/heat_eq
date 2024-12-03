// rendering.cpp

#include "rendering.h"
#include "constants.h"
#include "utility.h"
#include <stdio.h>

void init_opengl(SimulationMode simulation_mode, GLFWwindow **window, GLuint *pbo, struct cudaGraphicsResource **cuda_pbo_resource)
{
    if (!glfwInit())
    {
        printf("Failed to initialize GLFW\n");
        exit(-1);
    }

    int window_width = WIDTH;
    int window_height = (simulation_mode == MODE_1D) ? 100 : HEIGHT;
    char title[256];
    sprintf(title, "CUDA Heat Equation - Width: %d Height: %d", WIDTH, HEIGHT);

    *window = glfwCreateWindow(window_width, window_height, title, NULL, NULL);
    if (!*window)
    {
        printf("Failed to create window\n");
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(*window);

    glewInit();

    glGenBuffers(1, pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);

    size_t pbo_size;
    if (simulation_mode == MODE_1D)
    {
        pbo_size = WIDTH * sizeof(uchar4);
    }
    else if (simulation_mode == MODE_2D)
    {
        pbo_size = WIDTH * HEIGHT * sizeof(uchar4);
    }
    else if (simulation_mode == MODE_3D)
    {
        pbo_size = WIDTH * HEIGHT * DEPTH * sizeof(uchar4);
    }

    glBufferData(GL_PIXEL_UNPACK_BUFFER, pbo_size, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(cuda_pbo_resource, *pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void render(GLFWwindow *window, SimulationMode simulation_mode, uchar4 *d_output, struct cudaGraphicsResource *cuda_pbo_resource, GLuint pbo)
{
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    size_t size;
    uchar4 *output;
    cudaGraphicsResourceGetMappedPointer((void **)&output, &size, cuda_pbo_resource);

    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    if (simulation_mode == MODE_1D)
    {
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