// simulation.h

#ifndef SIMULATION_H
#define SIMULATION_H

#include "kernels.h"

void init_simulation(float **d_u0, float **d_u1, size_t *size, SimulationMode mode);
void reset_simulation(float *d_u0, float *d_u1, size_t size);

void simulate(float *d_u0, float *d_u1, size_t size, uchar4 *d_output, SimulationMode mode, BoundaryCondition boundary_condition);

void add_heat_launcher(float *d_u0, SimulationMode simulation_mode, double xpos, double ypos, int window_width, int window_height);

#endif // SIMULATION_H