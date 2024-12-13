# Heat Equation Simulation Project

The main showcase of this project is CUDA and OpenGL interop, where we leverage high performance from CUDA and get real-time rendering via OpenGL.

## Math Background

The heat equation models how heat diffuses through a medium over time. Here's an overview of the 2D and 3D cases, including their discretization and boundary conditions.

The heat equation is a partial differential equation (PDE) that describes the distribution of heat (or variation in temperature) in a given region over time. 

### Different General Forms of the Heat Equation based on Spacial Dimensions

All of these are in continuous form, and we are assuming non-homogenous heat sources.
* In the 1D case, the heat equation is given by:

  $$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2} + Q(x, t) $$

  * $u(x, t)$: Temperature at position ( x ) and time ( t ).
  * $Q(x, t)$: Heat source term (e.g., from mouse interactions in the simulation).
  * $\alpha $: Thermal diffusivity constant.


* In the 2D case, the heat equation is given by:

  $$\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) + Q(x, y, t) $$

  * $u(x, y, t)$: Temperature at point ( (x, y) ) and time ( t ).
  * $Q(x, y, t)$: Heat source term (e.g., from mouse interactions in the simulation).
  * $\alpha $: Thermal diffusivity constant.

  We use a flattened vector to store the grid points, as it simplifies computations compared to using a 2D matrix for $x$ and $y$ coordinates.

* In the 3D case, the heat equation is given by:

  $$\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right) + Q(x, y, z, t) $$

  * $u(x, y, z, t)$: Temperature at point ( (x, y, z) ) and time ( t ).
  * $Q(x, y, z, t)$: Heat source term (e.g., from mouse interactions in the simulation).
  * $\alpha $: Thermal diffusivity constant.

  We use a flattened vector to store the grid points, as it simplifies computations compared to using a 3D matrix for $x$, $y$, and $z$ coordinates.

### Discretization
Using finite difference methods on a grid with spacings for discretization of space with $\Delta x$, $\Delta y$, $\Delta z$ depending on the dimension and time with $\Delta t$. This allows us to approximate the derivatives and numerically solve the heat equation.

Spatial Derivatives:

Approximate second derivatives:

* $\frac{\partial^2 u}{\partial x^2}$: Second derivative with respect to $x$.
  $$\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{(\Delta x)^2} $$

* $\frac{\partial^2 u}{\partial y^2}$: Second derivative with respect to $y$.
  $$\frac{\partial^2 u}{\partial y^2} \approx \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{(\Delta y)^2} $$

* $\frac{\partial^2 u}{\partial z^2}$: Second derivative with respect to $z$.
  $$\frac{\partial^2 u}{\partial z^2} \approx \frac{u_{i,j,k+1} - 2u_{i,j,k} + u_{i,j,k-1}}{(\Delta z)^2} $$

* $\frac{\partial u}{\partial t}$: First derivative with respect to time.
This is a forward difference in time that will propagate the heat equation forward in time.
  $$\frac{\partial u}{\partial t} \approx \frac{u^{n+1}{i,j} - u^{n}{i,j}}{\Delta t} $$


* Discretized Heat Equation - In 3D:

  $$\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right) + Q(x, y, z, t) \implies $$

  $$\frac{u^{n+1}_{i,j,k} - u^{n}_{i,j,k}}{\Delta t} = \alpha \left( \frac{u_{i+1,j,k}^{n} - 2u_{i,j,k}^{n} + u_{i-1,j,k}^{n}}{(\Delta x)^2} + \frac{u_{i,j+1,k}^{n} - 2u_{i,j,k}^{n} + u_{i,j-1,k}^{n}}{(\Delta y)^2} + \frac{u_{i,j,k+1}^{n} - 2u_{i,j,k}^{n} + u_{i,j,k-1}^{n}}{(\Delta z)^2} \right) + Q_{i,j,k}^{n} $$

* Solving for $u^{n+1}_{i,j,k}$: The next temperature at point $(i, j, k)$.

  $$u^{n+1}_{i,j,k} = u^{n}_{i,j,k} + \Delta t \times \alpha \left( \frac{u_{i+1,j,k}^{n} - 2u_{i,j,k}^{n} + u_{i-1,j,k}^{n}}{(\Delta x)^2} + \frac{u_{i,j+1,k}^{n} - 2u_{i,j,k}^{n} + u_{i,j-1,k}^{n}}{(\Delta y)^2} + \frac{u_{i,j,k+1}^{n} - 2u_{i,j,k}^{n} + u_{i,j,k-1}^{n}}{(\Delta z)^2} \right) + \Delta t \times Q_{i,j,k}^{n} $$

### Boundary Conditions

Boundary conditions are essential to solving the heat equation. In this simulation, we use two types of boundary conditions:
While each one can be different and separate, for the sake of simplicity we are setting them all to the same value. 

* Dirichlet Boundary Condition:

  In the continuous form, this sets the temperature at the boundaries to a fixed value, which looks like:

  $$u(x, 0, 0, t) = u_{\text{boundary}}, \quad u(x, N_y, N_z, t) = u_{\text{boundary}}, $$
  $$\quad u(0, y, 0, t) = u_{\text{boundary}}, \quad u(N_x, y, N_z, t) = u_{\text{boundary}} $$
  $$\quad u(0, 0, z, t) = u_{\text{boundary}}, \quad u(N_x, N_y, z, t) = u_{\text{boundary}} $$

  And in the discretized form it will look the same, just we will index those specified, $0$ or $N$ values.
  Also, while the Dirichlet boundary condition is hardcoded to be $0.0f$, it can be changed to any value.

* Neumann Boundary Condition:

  In the continuous form, this sets the derivative of the temperature at the boundaries to a fixed value, which looks like:

  $$\left( \frac{\partial u}{\partial x} \right)_{\text{boundary}} = \text{HEAT\_SOURCE}, $$
  $$\left( \frac{\partial u}{\partial y} \right)_{\text{boundary}} = \text{HEAT\_SOURCE}, $$
  $$\left( \frac{\partial u}{\partial z} \right)_{\text{boundary}} = \text{HEAT\_SOURCE}  $$

  Set the derivative at the boundaries using finite differences:

    $$\frac{u_{0,j}^{n} - u_{1,j}^{n}}{2 \Delta x} = \left( \frac{\partial u}{\partial x} \right)_{\text{boundary}} $$

    Rearranged for implementation:

    For the left boundary ( \( i = 0 \) ):

    $$u_{0,j}^{n} = u_{1,j}^{n} - 2 \Delta x \left( \frac{\partial u}{\partial x} \right)_{\text{boundary}} $$

    For the rest&mdash;right, left, top, and bottom, and even the front and back in the 3D case&mdash;the same idea is applied.

### Non-Homogenous via. Adding Heat

As talked about in the different cases I have different $Q(\dots)$ terms. This is the heat source term, representing the amount of heat added at a given spacial position respective of the dimension I am simulating.

All this is going to be doing is $$Q(x, t) = Q(x, y, t) = Q(x, y, z, t)= \text{HEAT\_SOURCE},$$
where $\text{HEAT\_SOURCE}$ is a constant value.

### Stability Considerations
* Time Step ($\Delta t$):
  Should be chosen based on the grid spacings to ensure numerical stability.

  In 3D that would look something like:

  $$\Delta t \leq \frac{1}{2 \alpha} \left( \frac{1}{(\Delta x)^{-2} + (\Delta y)^{-2} + (\Delta z)^{-2}} \right) $$

  In Code:

  `#define TIME_STEP` is calculated using the grid spacings and diffusivity, with the said inequality; and set it to the upper bound of the inequality.

### Code Implementation Highlights
* Flattened Vectors:

  The grid is stored in a 1D array for computational efficiency.

  However, for the 2D and 3D cases, we will use a flattened vector to store the grid points, as it simplifies computations compared to using a 2D matrix for $x$ and $y$ coordinates, or a 3D matrix for $x$, $y$, and $z$ coordinates.
  
  Indexing can be done with a helper functions like IDX_2D and IDX_3D, but it can also be hard coded in the kernel to move indices by some constant values depending on how many points are in each dimension.
* Utilizing CUDA Kernels:
  They implement said discretization above and are used to simulate the heat equation over time.
  It does parallel computations for the different spacial grid points, as over time is a very sequential process, so we want to parallelize the spacial computations.
  - `heat_kernel_1d_sim`
  - `heat_kernel_2d_sim` 
  - `heat_kernel_3d_sim`

* Boundary Conditions in Code:
  Conditional statements check if a thread is at a boundary.
  Apply the appropriate boundary condition during each update. 
  
  **Note** that there will be thread divergence in the kernel due to this.

* Heat Source Term (Q):
  User interactions (e.g., mouse clicks) modify the temperature grid.
  Implemented in more CUDA kernels called:
  - `add_heat_kernel_1d`
  - `add_heat_kernel_2d` 
  - `add_heat_kernel_3d`
  
  **Note** given we are updating just a small portion of the grid, we wont be using shared memory for this, nor will we do work on the whole grid of points.

## How to Compile and Run

### Prerequisites
Before compiling the project, ensure you have the following installed on your Linux machine:

* CUDA Toolkit: Required for CUDA development.
* OpenGL Libraries: For rendering the simulation.
* GLEW: The OpenGL Extension Wrangler Library.
* GLFW: Open Source, multi-platform library for OpenGL.

You can install OpenGL, GLEW, and GLFW using your package manager. For Ubuntu/Debian-based systems:

```bash
sudo apt-get install build-essential libglfw3-dev libglew-dev libgl1-mesa-dev libglu1-mesa-dev
```

Also note that if you have multiple display devices, like a GPU and an APU/integrated graphics, you might need to set the default display device to the GPU. I am not going into how that can be done, but it is something to keep in mind.

### Compiling the Program
Use the provided `Makefile` to compile the project. Open a terminal in the project directory and run:

```bash
make
```
This will compile all the source files and generate an executable named `cuda_heat_equation`.

Note: If you encounter errors related to missing CUDA headers (e.g., `cuda_runtime.h`), ensure that the CUDA include directory is in your compiler's include path. The `Makefile` should handle this, but if issues persist, you might need to modify the `NVCC_FLAGS` in the `Makefile` to include the correct CUDA include path:


```bash
NVCC_FLAGS := --use_fast_math -O3 -gencode arch=compute_86,code=sm_86 -I/usr/local/cuda/include -Xptxas -v
```

Replace include with the path to your CUDA installation if it's different, and update the architecture to the GPU you have, as the one in the `Makefile` is for a RTX 30 series card.

### Running the Program

Execute the program with:

```bash
./cuda_heat_equation
```

By default, the simulation runs in 3D mode with Dirichlet boundary conditions &emdash; equavalent to `./cuda_heat_equation -m 3d -b d`. However, these can both be changed at launch using command-line arguments, or while running the simulation interactively.


* Command-Line

  The program accepts the following command-line arguments to set the simulation mode and boundary conditions:

  ```bash
  ./cuda_heat_equation -m <1d|2d|3d> -b <D|N> -d <max_time> <percent_add_heat>
  ```

  * -m <dimension>: Set the simulation mode.
    * 1d: 1-Dimensional simulation.
    * 2d: 2-Dimensional simulation.
    * 3d: 3-Dimensional simulation.

  * -b <condition_type>: Set the boundary condition type.
    * D: Dirichlet boundary conditions.
    * N: Neumann boundary conditions.

  * -d <max_time> <percent_add_heat>: 
  
    Set the maximum simulation time and the percentage of heat added per time step. These can also be left empty, and they will default to 100 and 40, respectively.
    * <max_time>: Maximum simulation time (integer greater than 0).
    * <percent_add_heat>: Percentage of heat to add (integer between 0 and 100).

  * More control:
    If you want to change the window/grid size, you can do so by changing the `WIDTH`, `HEIGHT`, and even the `DEPTH` which will be the number of "slices" in the z direction. This can be done
    in the `cuda_heat_equation.cu` file at the top.

* Interactive Mode
  
    Once the simulation is running, you can use keyboard and mouse controls to interact with the simulation.
  
    * Keyboard Controls:
      * 1: Switch to 1D simulation.
      * 2: Switch to 2D simulation.
      * B: Toggle boundary conditions between Dirichlet and Neumann.
      * R: Reset the simulation.
      * X: Exit the simulation.
  
    * Mouse Controls:
      * Left Mouse Button: Click and drag to add heat.

### Cleaning the Build

To clean up compiled files and the executable, run:

```bash
make clean
```

This removes object files and the cuda_heat_equation executable

## Ways Performance Gains Were Made

* Using the GPU to do both the computation of the simulation of each time step and the rendering of the simulation allows us to not transfer large amounts of data over the PCIe bus.

* Use of shared memory. The whole process when discretizing and simulating the next time step needs not just a given space's previous heat value but the neighboring values too. 

* The 3D kernel was further optimized by looking at Paulius Micikevicius's white paper given below, and Ashwin Srinath's code implementation of said white paper. The basic idea was to reduce on register counts by stepping through the flattened 3D array in slices.


## Possible Future Development: 

### Better color gradients in general. 
An example of what someone might want to try is a gradient based on MAGMA: 
Where the ranges could look like:
```python
magma_colormap = [
    [252, 253, 191],
    [254, 176, 120],
    [241, 96, 93],
    [183, 55, 121],
    [114, 31, 129],
    [44, 17, 95],
    [0, 0, 4]
]
```
Or other common gradients and some n-number of colors can be found here:
https://waldyrious.net/viridis-palette-generator/

Note that currently it is being I just have a low and high color and scaling the temperature based on that. It is hard to do that in general. Possibly there might be some use of sin/cos??? to combine them better. However, I do not know what would be the least computationally intensive way to do this.


## Sources Used

- [Cuda Samples by NVIDIA](https://github.com/NVIDIA/cuda-samples)
  - [Fluid Rendering with OpenGL](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/fluidsGL)
  - [Example 3D Cos-Sin Graph with OpenGL](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleGL)
- [Cuda Heat Equation](https://enccs.github.io/OpenACC-CUDA-beginners/2.02_cuda-heat-equation/)
- [3D Finite Difference Computation on GPUs using CUDA, by Paulius Micikevicius](https://developer.download.nvidia.com/CUDA/CUDA_Zone/papers/gpu_3dfd_rev.pdf)
  - [Implemntation done by Ashwin Srinath](https://github.com/shwina/cuda_3Dheat)
- ChatGPT o1-preview
- ChatGPT 4o