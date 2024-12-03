// rendering.h

#ifndef RENDERING_H
#define RENDERING_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glu.h>
#include <cuda_gl_interop.h>
#include "kernels.h"

void init_opengl(SimulationMode simulation_mode, GLFWwindow **window, GLuint *pbo, struct cudaGraphicsResource **cuda_pbo_resource);

void render(GLFWwindow *window, SimulationMode simulation_mode, uchar4 *d_output, struct cudaGraphicsResource *cuda_pbo_resource, GLuint pbo);

#endif // RENDERING_H