// constants.h

#ifndef CONSTANTS_H
#define CONSTANTS_H

// Simulation settings
#define WIDTH 1000
#define HEIGHT 1000
#define DEPTH 100
#define HEAT_SOURCE 5.0f
#define DX 1.0f
#define DY 1.0f
#define DZ 1.0f

#define DX2 (DX * DX)
#define DY2 (DY * DY)
#define DZ2 (DZ * DZ)

#define HEAT_RADIUS 5
#define DIFFUSIVITY 1.0f

#define TIME_STEP (DX2 * DY2 * DZ2 / (2 * DIFFUSIVITY * (DX2 + DY2 + DZ2)))

// CUDA block size
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_Z 16

#define SCALE_FACTOR_3D 0.8f

// Clamp function
#define HEAT_MAX_CLAMP 1.0f
#define HEAT_MIN_CLAMP 0.0f
// #define clamp(x) (x < HEAT_MIN_CLAMP ? HEAT_MIN_CLAMP : (x > HEAT_MAX_CLAMP ? HEAT_MAX_CLAMP : x))

#endif // CONSTANTS_H