#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <string.h>
#include <GLFW/glfw3.h>
#include <math.h>

#define WIDTH 1000
#define HEIGHT 1000
#define TIME_STEP 0.25f
#define DIFFUSIVITY 1.0f
#define HEAT_SOURCE 5.0f
#define DX 1.0f
#define DY 1.0f
#define HEAT_RADIUS 5

// Host variables
GLuint pbo;
GLFWwindow *window;

// Global variables for FPS calculation
double lastTime = 0.0;
int nbFrames = 0;
double fps = 0.0;

// Mouse state
bool is_mouse_pressed = false;

// Simulation modes
enum SimulationMode
{
    MODE_1D,
    MODE_2D
};
SimulationMode simulation_mode = MODE_2D;

// Boundary conditions
enum BoundaryCondition
{
    DIRICHLET,
    NEUMANN
};
BoundaryCondition boundary_condition = DIRICHLET;

// Debug Mode - For Profiling
bool debug_mode = false;
int MAX_TIME_STEPS = 100;
int PERCENT_ADD_HEAT_CHANCE = 40;

// Function prototypes
void init_simulation();
void init_opengl();
void update_sim_render();
void reset_simulation();
void keyboard_callback(GLFWwindow *window, int key, int scancode, int action, int mods);
void cursor_position_callback(GLFWwindow *window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);

// Host variables for simulation
float *h_u0, *h_u1;
struct uchar4 {
    unsigned char x, y, z, w;
};
uchar4 *h_output;

// Color Functions
uchar4 gradient_scaling(float standard_heat_value);

// Clamp function
#define HEAT_MAX_CLAMP 1.0f
#define HEAT_MIN_CLAMP 0.0f
#define clamp(x) (x < HEAT_MIN_CLAMP ? HEAT_MIN_CLAMP : (x > HEAT_MAX_CLAMP ? HEAT_MAX_CLAMP : x))

uchar4 gradient_scaling(float standard_heat_value)
{
    unsigned char r = (unsigned char)(255 * clamp(standard_heat_value / HEAT_SOURCE));
    unsigned char g = 0;
    unsigned char b = (unsigned char)(255 * (1 - clamp(standard_heat_value / HEAT_SOURCE)));
    return (uchar4){r, g, b, 255};
}

// Heat kernel for 1D simulation
void heat_kernel_1d(float *u0, float *u1, uchar4 *output, int width, float dt, float dx2, float a, BoundaryCondition boundary_condition)
{
    for (int x = 0; x < width; ++x)
    {
        if (x > 0 && x < width - 1)
        {
            float u_center = u0[x];
            float u_left = u0[x - 1];
            float u_right = u0[x + 1];

            float laplacian = (u_left - 2 * u_center + u_right) / dx2;

            u1[x] = u_center + a * dt * laplacian;
        }
        else
        {
            switch (boundary_condition)
            {
            case DIRICHLET:
                u1[x] = 0.0f;
                break;
            case NEUMANN:
                if (x == 0)
                    u1[x] = u1[x + 1] + HEAT_SOURCE * dx2;
                else if (x == width - 1)
                    u1[x] = u1[x - 1] + HEAT_SOURCE * dx2;
                break;
            }
        }
        unsigned char color = (unsigned char)(255 * clamp(u1[x] / HEAT_SOURCE));
        output[x] = (uchar4){color, 0, 255 - color, 255};
    }
}

// Heat kernel for 2D simulation
void heat_kernel_2d(float *u0, float *u1, uchar4 *output, int width, int height, float dt, float dx2, float dy2, float a, BoundaryCondition boundary_condition)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;
            if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
            {
                float u_center = u0[idx];
                float u_left = u0[idx - 1];
                float u_right = u0[idx + 1];
                float u_down = u0[idx - width];
                float u_up = u0[idx + width];

                float laplacian = (u_left - 2 * u_center + u_right) / dx2 + (u_up - 2 * u_center + u_down) / dy2;

                u1[idx] = u_center + a * dt * laplacian;
            }
            else
            {
                switch (boundary_condition)
                {
                case DIRICHLET:
                    u1[idx] = 0.0f;
                    break;
                case NEUMANN:
                    if (x == 0)
                        u1[idx] = u1[idx + 1] + HEAT_SOURCE * dx2;
                    else if (x == width - 1)
                        u1[idx] = u1[idx - 1] + HEAT_SOURCE * dx2;
                    else if (y == 0)
                        u1[idx] = u1[idx + width] + HEAT_SOURCE * dy2;
                    else if (y == height - 1)
                        u1[idx] = u1[idx - width] + HEAT_SOURCE * dy2;
                    break;
                }
            }
            unsigned char color = (unsigned char)(255 * clamp(u1[idx] / HEAT_SOURCE));
            output[idx] = (uchar4){color, 0, 255 - color, 255};
        }
    }
}

// Add heat kernel for 1D simulation
void add_heat_kernel_1d(float *u, int width, int x)
{
    for (int tx = -HEAT_RADIUS; tx <= HEAT_RADIUS; ++tx)
    {
        int idx = x + tx;
        if (idx >= 0 && idx < width)
        {
            u[idx] += HEAT_SOURCE;
        }
    }
}

// Add heat kernel for 2D simulation
void add_heat_kernel_2d(float *u, int width, int height, int cx, int cy)
{
    for (int ty = -HEAT_RADIUS; ty <= HEAT_RADIUS; ++ty)
    {
        for (int tx = -HEAT_RADIUS; tx <= HEAT_RADIUS; ++tx)
        {
            int x = cx + tx;
            int y = cy + ty;
            if (x >= 0 && x < width && y >= 0 && y < height)
            {
                if (tx * tx + ty * ty <= HEAT_RADIUS * HEAT_RADIUS)
                {
                    int idx = y * width + x;
                    u[idx] += HEAT_SOURCE;
                }
            }
        }
    }
}

// Initialize the simulation
void init_simulation()
{
    size_t size;
    if (simulation_mode == MODE_1D)
    {
        size = WIDTH * sizeof(float);
    }
    else
    {
        size = WIDTH * HEIGHT * sizeof(float);
    }

    h_u0 = (float *)calloc(size, sizeof(float));
    h_u1 = (float *)calloc(size, sizeof(float));
    h_output = (uchar4 *)calloc(size, sizeof(uchar4));
}

// Initialize OpenGL
void init_opengl()
{
    if (!glfwInit())
    {
        printf("Failed to initialize GLFW\n");
        exit(-1);
    }

    int window_width = WIDTH;
    int window_height = (simulation_mode == MODE_1D) ? 100 : HEIGHT;

    window = glfwCreateWindow(window_width, window_height, "CUDA Heat Equation", NULL, NULL);
    if (!window)
    {
        printf("Failed to create window\n");
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(window);

    glewInit();

    // Create PBO
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    size_t pbo_size = (simulation_mode == MODE_1D) ? WIDTH * sizeof(uchar4) : WIDTH * HEIGHT * sizeof(uchar4);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, pbo_size, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

// Render simulation
void update_sim_render()
{
    float dx2 = DX * DX;
    float dy2 = DY * DY;

    if (simulation_mode == MODE_1D)
    {
        heat_kernel_1d(h_u0, h_u1, h_output, WIDTH, TIME_STEP, dx2, DIFFUSIVITY, boundary_condition);
    }
    else if (simulation_mode == MODE_2D)
    {
        heat_kernel_2d(h_u0, h_u1, h_output, WIDTH, HEIGHT, TIME_STEP, dx2, dy2, DIFFUSIVITY, boundary_condition);
    }

    // Swap pointers
    float *temp = h_u0;
    h_u0 = h_u1;
    h_u1 = temp;

    // FPS calculation
    double currentTime = glfwGetTime();
    nbFrames++;
    if (currentTime - lastTime >= 1.0)
    {
        fps = double(nbFrames) / (currentTime - lastTime);
        nbFrames = 0;
        lastTime += 1.0;

        // Update window title
        char title[256];
        sprintf(title, "Heat Equation - Width: %d Height: %d FPS: %.2f", WIDTH, HEIGHT, fps);
        glfwSetWindowTitle(window, title);
    }

    // Draw pixels
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, WIDTH * HEIGHT * sizeof(uchar4), h_output);
    if (simulation_mode == MODE_1D)
    {
        for (int i = 0; i < 20; i++)
        {
            glRasterPos2f(-1.0f, -1.0f + i * 0.02f);
            glDrawPixels(WIDTH, 1, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        }
    }
    else
    {
        glRasterPos2i(-1, -1);
        glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glfwSwapBuffers(window);
}

// Reset simulation
void reset_simulation()
{
    size_t size;
    if (simulation_mode == MODE_1D)
    {
        size = WIDTH * sizeof(float);
    }
    else
    {
        size = WIDTH * HEIGHT * sizeof(float);
    }
    memset(h_u0, 0, size);
    memset(h_u1, 0, size);
}

// Keyboard callback
void keyboard_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_1:
            simulation_mode = MODE_1D;
            printf("Switched to 1D simulation\n");
            glDeleteBuffers(1, &pbo);
            free(h_u0);
            free(h_u1);
            glfwDestroyWindow(window);
            init_opengl();
            init_simulation();

            glfwSetKeyCallback(window, keyboard_callback);
            glfwSetMouseButtonCallback(window, mouse_button_callback);
            glfwSetCursorPosCallback(window, cursor_position_callback);

            break;
        case GLFW_KEY_2:
            simulation_mode = MODE_2D;
            printf("Switched to 2D simulation\n");
            glDeleteBuffers(1, &pbo);
            free(h_u0);
            free(h_u1);
            glfwDestroyWindow(window);
            init_opengl();
            init_simulation();

            glfwSetKeyCallback(window, keyboard_callback);
            glfwSetMouseButtonCallback(window, mouse_button_callback);
            glfwSetCursorPosCallback(window, cursor_position_callback);

            break;
        case GLFW_KEY_B:
            boundary_condition = (boundary_condition == DIRICHLET) ? NEUMANN : DIRICHLET;
            printf("Switched boundary condition to %s\n",
                   (boundary_condition == DIRICHLET) ? "Dirichlet" : "Neumann");
            break;
        case GLFW_KEY_R:
            reset_simulation();
            printf("Simulation reset\n");
            break;
        case GLFW_KEY_X:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            printf("Exiting simulation\n");

            // Cleanup
            glDeleteBuffers(1, &pbo);
            free(h_u0);
            free(h_u1);
            glfwDestroyWindow(window);
            glfwTerminate();
            exit(0);
            break;
        default:
            break;
        }
    }
}

// Cursor position callback
void cursor_position_callback(GLFWwindow *window, double xpos, double ypos)
{
    if (is_mouse_pressed)
    {
        int fb_width, fb_height;
        glfwGetFramebufferSize(window, &fb_width, &fb_height);

        if (simulation_mode == MODE_1D)
        {
            int x = (int)(xpos * WIDTH / fb_width);
            add_heat_kernel_1d(h_u0, WIDTH, x);
        }
        else
        {
            int x = (int)(xpos * WIDTH / fb_width);
            int y = (int)((fb_height - ypos) * HEIGHT / fb_height); // Invert y-axis
            add_heat_kernel_2d(h_u0, WIDTH, HEIGHT, x, y);
        }
    }
}

// Mouse button callback
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            is_mouse_pressed = true;
        }
        else if (action == GLFW_RELEASE)
        {
            is_mouse_pressed = false;
        }
    }
}

// Main function
int main(int argc, char **argv)
{
    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)
        {
            if (strcmp(argv[i + 1], "1d") == 0)
            {
                simulation_mode = MODE_1D;
            }
            else if (strcmp(argv[i + 1], "2d") == 0)
            {
                simulation_mode = MODE_2D;
            }
            ++i;
        }
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc)
        {
            if (strcmp(argv[i + 1], "d") == 0)
            {
                boundary_condition = DIRICHLET;
            }
            else if (strcmp(argv[i + 1], "n") == 0)
            {
                boundary_condition = NEUMANN;
            }
            ++i;
        }
        else if (strcmp(argv[i], "-d") == 0)
        {
            debug_mode = true;
            if (i + 2 < argc)
            {
                MAX_TIME_STEPS = atoi(argv[i + 1]);
                PERCENT_ADD_HEAT_CHANCE = atoi(argv[i + 2]);
                if (MAX_TIME_STEPS < 0)
                {
                    MAX_TIME_STEPS = 100;
                }
                if (PERCENT_ADD_HEAT_CHANCE < 0 || PERCENT_ADD_HEAT_CHANCE > 100)
                {
                    PERCENT_ADD_HEAT_CHANCE = 40;
                }
                i += 2;
            }
        }
    }

    init_opengl();
    init_simulation();

    glfwSetKeyCallback(window, keyboard_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    if (debug_mode)
    {
        for (int i = 0; i < MAX_TIME_STEPS; i++)
        {
            update_sim_render();
            if (rand() % 100 < PERCENT_ADD_HEAT_CHANCE)
            {
                int x = rand() % WIDTH;
                int y = rand() % HEIGHT;
                add_heat_kernel_2d(h_u0, WIDTH, HEIGHT, x, y);
            }
        }
    }
    else
    {
        while (window && !glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            update_sim_render();
        }

    }
    
    // Cleanup
    glDeleteBuffers(1, &pbo);
    free(h_u0);
    free(h_u1);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
