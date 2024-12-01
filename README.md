# Heat Equation Simulation Project

The main showcase of this project is CUDA and OpenGL interop, where we leverage high performance from CUDA and get real-time rendering via OpenGL.

## Math Background

The heat equation is a partial differential equation (PDE) that describes the distribution of heat (or variation in temperature) in a given region over time. 

### 1D General Form of the Heat Equation

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
$$

where:
- $ u(x, t) $ is the temperature distribution function, vector.
- $ t $ is time, scalar.
- $ x $ is the spatial coordinate, vector.
- $ \alpha $ is the thermal diffusivity of the material, scalar.

### 2D General Form of the Heat Equation

$$
\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$

where:
- $ u(x, y, t) $ is the temperature distribution function, flattened vector.
- $ t $ is time, scalar.
- $ x $ and $ y $ are the spatial coordinates, flattened vector.
- $ \alpha $ is the thermal diffusivity of the material, scalar.

We use a flattened vector to store the grid points, as it simplifies computations compared to using a 2D matrix for $x$ and $y$ coordinates.

### Boundary Conditions

Boundary conditions are essential to solving the heat equation. In this simulation, we use two types of boundary conditions:
While each one can be differnet and seperate, for the sake of simplicity we are setting them all to the same value. 

1. **Dirichlet Boundary Condition:**
    - Specifies the value of the function $u(x,t)$ on the boundary.
    - In the provided code we are doing:
        - in 1D: $$u(0, t) = u(L, t) = q_0$$
        - In 2D: $$ u(0, y, t) = u(L, y, t) = u(x, 0, t) = u(x, H, t) = q_0 $$

2. **Neumann Boundary Condition:**
    - Specifies the value of the derivative of the function on the boundary.
    - In the provided code we are doing:
        - in 1D: 
        $$ \frac{\partial u}{\partial x}(0, t) = \frac{\partial u}{\partial x}(L, t) = q_1 $$
        - In 2D: $$ \frac{\partial u}{\partial x}(0, y, t) = \frac{\partial u}{\partial x}(L, y, t) = \frac{\partial u}{\partial y}(x, 0, t) = \frac{\partial u}{\partial y}(x, H, t) = q_1 $$


### Non-Homogenous via. Adding Heat

$$
\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) + Q(x, y, t)
$$

where:
- $ Q(x, y, t) $ is the heat source term, representing the amount of heat added at position $(x, y)$ and time $t$.

In our simulation, $Q(x, y, t)$ is determined by the mouse interactions, where the left mouse button click and drag adds heat to the grid points dynamically.


### Discretization

To solve the heat equation numerically, we discretize the spatial and temporal domains. Using finite difference methods, we approximate the derivatives with difference equations.
This is a bit more in the weeds, so I will not go through this as much. However if you want to read more about it go through: [Cuda Heat Equation](https://enccs.github.io/OpenACC-CUDA-beginners/2.02_cuda-heat-equation/).

## How to Compile and Run

Use the provided makefile. All you need to do is run `make`. 
For running the program, use `./cuda_heat_equation`.

This function processes input arguments to set up boundary conditions for a simulation.

Parameters:
- `-m <dimention>`: Specifies the mode of the simulation. Acceptable values are `1d` or `2d`.
- `-b <condition_type>`: Specifies the boundary condition type. Acceptable values are `d` for Dirichlet or `n` for Neumann boundary conditions.
- `-d <max_time> <percent_add_heat>`: Specifies that the program is going to be in debugging mode. Acceptable values are integers for both. Make sure percent adding heat is between 0 and 100, and time is greater than 0. If not defaulted to 100 and 40

Usage:
- `-m 1d` or `-m 2d` to set the simulation mode to 1-dimensional or 2-dimensional, respectively.
- `-b d` or `-b n` to set the boundary condition to Dirichlet or Neumann, respectively.
- `-d 100 40`

By default if you call: 
- `./cuda_heat_equation` it is the same as `./cuda_heat_equation -m 2d -b d`

### Interactive Functionalities

#### Keyboard Controls
- **1**: Switch to 1D simulation.
- **2**: Switch to 2D simulation.
- **B**: Toggle boundary conditions between Dirichlet and Neumann.
- **R**: Reset the simulation.

#### Mouse Controls
- **Left Mouse Button**: Click and drag to add heat.

### Boundary Conditions

- **Dirichlet Conditions**: Hardcoded to be $ 0.0f $
- **Neumann Conditions**: Adds `HEAT_SOURCE` (5.0f) amount of heat over time.

## Ways Performance Gains Were Made




## Sources Used

- [Cuda Samples by NVIDIA](https://github.com/NVIDIA/cuda-samples)
  - [Fluid Rendering with OpenGL](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/fluidsGL)
  - [Example 3D Cos-Sin Graph with OpenGL](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleGL)
- [Cuda Heat Equation](https://enccs.github.io/OpenACC-CUDA-beginners/2.02_cuda-heat-equation/)
- ChatGPT o1-preview
- ChatGPT 4o