#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <math.h>
#include <vector>
#include <cstring>
#include <unistd.h>
#include <getopt.h>

#define WIDTH 1000
#define HEIGHT 1000
#define DEPTH 100
#define TIME_STEP 0.25f
#define DIFFUSIVITY 1.0f
#define HEAT_SOURCE 5.0f
#define DX 1.0f
#define DY 1.0f
#define DZ 1.0f
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
    MODE_2D,
    MODE_3D
};
SimulationMode simulation_mode = MODE_3D;

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
void add_heat_launcher(double xpos, double ypos);

// Host variables for simulation
std::vector<float> h_u0, h_u1;
struct uchar4
{
    unsigned char x, y, z, w;
};
std::vector<uchar4> h_output;

// Color Functions
void gradient_scaling(float heat_value, uchar4 *out_color);

// Clamp function
#define HEAT_MAX_CLAMP 1.0f
#define HEAT_MIN_CLAMP 0.0f
#define clamp(x) (x < HEAT_MIN_CLAMP ? HEAT_MIN_CLAMP : (x > HEAT_MAX_CLAMP ? HEAT_MAX_CLAMP : x))

void gradient_scaling(float heat_value, uchar4* out_color)
{
    // Gradient Set Up:
    float t = clamp(heat_value / HEAT_SOURCE);

    // Default alpha
    unsigned char a = 255;

    // Adjust alpha for 3D mode
    if (simulation_mode == MODE_3D) {
        a = (unsigned char) (255.0f / DEPTH * 4.0f);
    }

#if 0
    // liner interpolation between rgb(0, 0, 255) and rgb(255, 0, 0)
    // const float R_low = 0.0f;
    const float R_high = 255.0f;
    // const float G_low = 0.0f;
    // const float G_high = 0.0f;
    const float B_low = 255.0f;
    const float B_high = 0.0f;
    
    // Perform linear interpolation
    unsigned char r = (unsigned char) (t * R_high);
    unsigned char g = (unsigned char) (0);
    unsigned char b = (unsigned char) (B_low + t * (B_high - B_low));

    // Write the result to the output pointer   
    *out_color = (uchar4){r, g, b, a};
#else
    // liner interpolation between rgb(232, 251, 90) and rgb(5, 34, 51)
    const float R_low = 232.0f;
    const float R_high = 5.0f;
    const float G_low = 251.0f;
    const float G_high = 34.0f;
    const float B_low = 90.0f;
    const float B_high = 51.0f;
    
    // Perform linear interpolation
    unsigned char r = (unsigned char) (R_low + t * (R_high - R_low));
    unsigned char g = (unsigned char) (G_low + t * (G_high - G_low));
    unsigned char b = (unsigned char) (B_low + t * (B_high - B_low));

    // Write the result to the output pointer
    *out_color = (uchar4){r, g, b, a};
#endif
}

// Heat kernel for 1D simulation
void heat_kernel_1d(const std::vector<float> &u0, std::vector<float> &u1, std::vector<uchar4> &output, int width, float dt, float dx2, float a, BoundaryCondition boundary_condition)
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
        gradient_scaling(u1[x], &output[x]);
    }
}

// Heat kernel for 2D simulation
void heat_kernel_2d(const std::vector<float> &u0, std::vector<float> &u1, std::vector<uchar4> &output, int width, int height, float dt, float dx2, float dy2, float a, BoundaryCondition boundary_condition)
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

                float laplacian = (u_left - 2 * u_center + u_right) / dx2 + 
                    (u_up - 2 * u_center + u_down) / dy2;

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
            gradient_scaling(u1[x], &output[x]);
        }
    }
}

// Heat kernel for 3D simulation
void heat_kernel_3d(const std::vector<float> &u0, std::vector<float> &u1, std::vector<uchar4> &output, int width, int height, int depth, float dt, float dx2, float dy2, float dz2, float a, BoundaryCondition boundary_condition)
{
    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int idx = z * width * height + y * width + x;
                if (x > 0 && x < width - 1 && y > 0 && y < height - 1 && z > 0 && z < depth - 1)
                {
                    float u_center = u0[idx];
                    float u_left = u0[idx - 1];
                    float u_right = u0[idx + 1];
                    float u_down = u0[idx - width];
                    float u_up = u0[idx + width];
                    float u_back = u0[idx - width * height];
                    float u_front = u0[idx + width * height];

                    float laplacian = (u_left - 2 * u_center + u_right) / dx2 + 
                        (u_up - 2 * u_center + u_down) / dy2 + 
                        (u_front - 2 * u_center + u_back) / dz2;

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
                        else if (z == 0)
                            u1[idx] = u1[idx + width * height] + HEAT_SOURCE * dz2;
                        else if (z == depth - 1)
                            u1[idx] = u1[idx - width * height] + HEAT_SOURCE * dz2;
                        break;
                    }
                }
                gradient_scaling(u1[x], &output[x]);
            }
        }
    }
}

// Add heat kernel for 1D simulation
void add_heat_kernel_1d(std::vector<float> &u, int width, int x)
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
void add_heat_kernel_2d(std::vector<float> &u, int width, int height, int cx, int cy)
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

// Add heat kernel for 3D simulation
void add_heat_kernel_3d(std::vector<float> &u, int width, int height, int depth, int cx, int cy, int cz)
{
    for (int tz = -HEAT_RADIUS; tz <= HEAT_RADIUS; ++tz)
    {
        for (int ty = -HEAT_RADIUS; ty <= HEAT_RADIUS; ++ty)
        {
            for (int tx = -HEAT_RADIUS; tx <= HEAT_RADIUS; ++tx)
            {
                int x = cx + tx;
                int y = cy + ty;
                int z = cz + tz;
                if (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth)
                {
                    if (tx * tx + ty * ty + tz * tz <= HEAT_RADIUS * HEAT_RADIUS)
                    {
                        int idx = z * width * height + y * width + x;
                        u[idx] += HEAT_SOURCE;
                    }
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
        size = WIDTH;
    }
    else if (simulation_mode == MODE_2D)
    {
        size = WIDTH * HEIGHT;
    }
    else if (simulation_mode == MODE_3D)
    {
        size = WIDTH * HEIGHT * DEPTH;
    }

    h_u0.resize(size, 0.0f);
    h_u1.resize(size, 0.0f);
    h_output.resize(size);
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

    window = glfwCreateWindow(window_width, window_height, "Heat Equation Simulation", NULL, NULL);
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
    if (simulation_mode == MODE_3D)
    {
        pbo_size = WIDTH * HEIGHT * DEPTH * sizeof(uchar4);
    }
    glBufferData(GL_PIXEL_UNPACK_BUFFER, pbo_size, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

// Render simulation
void update_sim_render()
{
    float dx2 = DX * DX;
    float dy2 = DY * DY;
    float dz2 = DZ * DZ;

    if (simulation_mode == MODE_1D)
    {
        heat_kernel_1d(h_u0, h_u1, h_output, WIDTH, TIME_STEP, dx2, DIFFUSIVITY, boundary_condition);
    }
    else if (simulation_mode == MODE_2D)
    {
        heat_kernel_2d(h_u0, h_u1, h_output, WIDTH, HEIGHT, TIME_STEP, dx2, dy2, DIFFUSIVITY, boundary_condition);
    }
    else if (simulation_mode == MODE_3D)
    {
        heat_kernel_3d(h_u0, h_u1, h_output, WIDTH, HEIGHT, DEPTH, TIME_STEP, dx2, dy2, dz2, DIFFUSIVITY, boundary_condition);
    }

    // Swap pointers
    std::swap(h_u0, h_u1);

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
    if (simulation_mode == MODE_1D)
    {
        glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, WIDTH * sizeof(uchar4), h_output.data());
        for (int i = 0; i < 20; i++)
        {
            glRasterPos2f(-1.0f, -1.0f + i * 0.02f);
            glDrawPixels(WIDTH, 1, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        }
    }
    else if (simulation_mode == MODE_2D)
    {
        glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, WIDTH * HEIGHT * sizeof(uchar4), h_output.data());
        glRasterPos2i(-1, -1);
        glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    }
    else if (simulation_mode == MODE_3D)
    {
        glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, WIDTH * HEIGHT * DEPTH * sizeof(uchar4), h_output.data());

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

        // Scale factor to reduce the size of each slice
        float scale_factor = 0.6f; // Adjust as needed

        // Starting position for the backmost layer
        float start_x = 0.0f;
        float start_y = window_height - (HEIGHT * scale_factor);

        // Draw each slice with offset
        for (int z = 0; z < DEPTH; ++z)
        {
            float offset_x = start_x + z * (WIDTH * scale_factor) / DEPTH;
            float offset_y = start_y - z * (HEIGHT * scale_factor) / DEPTH;

            glRasterPos2f(offset_x, offset_y);

            // Set pixel zoom for scaling
            glPixelZoom(scale_factor, scale_factor);

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

// Reset simulation
void reset_simulation()
{
    std::fill(h_u0.begin(), h_u0.end(), 0.0f);
    std::fill(h_u1.begin(), h_u1.end(), 0.0f);
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
            h_u0.clear();
            h_u1.clear();
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
            h_u0.clear();
            h_u1.clear();
            glfwDestroyWindow(window);
            init_opengl();
            init_simulation();

            glfwSetKeyCallback(window, keyboard_callback);
            glfwSetMouseButtonCallback(window, mouse_button_callback);
            glfwSetCursorPosCallback(window, cursor_position_callback);

            break;
        case GLFW_KEY_3:
            simulation_mode = MODE_3D;
            printf("Switched to 3D simulation\n");
            glDeleteBuffers(1, &pbo);
            h_u0.clear();
            h_u1.clear();
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
            h_u0.clear();
            h_u1.clear();
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
        add_heat_launcher(xpos, ypos);
    }
}

void add_heat_launcher(double xpos, double ypos)
{

    int fb_width, fb_height;
    glfwGetFramebufferSize(window, &fb_width, &fb_height);

    if (simulation_mode == MODE_1D)
    {
        int x = (int)(xpos * WIDTH / fb_width);
        add_heat_kernel_1d(h_u0, WIDTH, x);
    }
    else if (simulation_mode == MODE_2D)
    {
        int x = (int)(xpos * WIDTH / fb_width);
        int y = (int)((fb_height - ypos) * HEIGHT / fb_height); // Invert y-axis
        add_heat_kernel_2d(h_u0, WIDTH, HEIGHT, x, y);
    }
    else if (simulation_mode == MODE_3D)
    {
        int x = (int)(xpos * WIDTH / fb_width);
        int y = (int)((fb_height - ypos) * HEIGHT / fb_height); // Invert y-axis
        int z = rand() % DEPTH;
        add_heat_kernel_3d(h_u0, WIDTH, HEIGHT, DEPTH, x, y, z);
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

// Print usage
void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("[options]:\n");
    printf("  -m <1d|2d|3d>         Simulation mode (default: 3d)\n");
    printf("  -b <D|N>              Boundary condition: D (Dirichlet), N (Neumann) (default: D)\n");
    printf("  -d [max_steps chance] Enable debug mode with optional max time steps (default: 100) and heat chance (default: 40)\n");
}

int main(int argc, char **argv) {
    int opt;
    while ((opt = getopt(argc, argv, "m:b:d::")) != -1) {
        switch (opt) {
            case 'm': // Simulation mode
                if (strcmp(optarg, "1d") == 0) {
                    simulation_mode = MODE_1D;
                } else if (strcmp(optarg, "2d") == 0) {
                    simulation_mode = MODE_2D;
                } else if (strcmp(optarg, "3d") == 0) {
                    simulation_mode = MODE_3D;
                } else {
                    fprintf(stderr, "Invalid simulation mode: %s\n", optarg);
                    print_usage(argv[0]);
                    return EXIT_FAILURE;
                }
                break;

            case 'b': // Boundary condition
                if (strcmp(optarg, "D") == 0) {
                    boundary_condition = DIRICHLET;
                } else if (strcmp(optarg, "N") == 0) {
                    boundary_condition = NEUMANN;
                } else {
                    fprintf(stderr, "Invalid boundary condition: %s\n", optarg);
                    print_usage(argv[0]);
                    return EXIT_FAILURE;
                }
                break;

            case 'd': // Debug mode
                debug_mode = true;
                if (optind < argc && argv[optind][0] != '-') {
                    MAX_TIME_STEPS = atoi(argv[optind++]);
                }
                if (optind < argc && argv[optind][0] != '-') {
                    PERCENT_ADD_HEAT_CHANCE = atoi(argv[optind++]);
                }

                // Validate debug parameters
                if (MAX_TIME_STEPS < 0) MAX_TIME_STEPS = 100;
                if (PERCENT_ADD_HEAT_CHANCE < 0 || PERCENT_ADD_HEAT_CHANCE > 100) {
                    PERCENT_ADD_HEAT_CHANCE = 40;
                }
                break;

            default: // Invalid option
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    // Print the parsed configuration
    printf("Simulation Mode: %s\n", simulation_mode == MODE_1D ? "1D" :
                                  simulation_mode == MODE_2D ? "2D" : "3D");
    printf("Boundary Condition: %s\n", boundary_condition == DIRICHLET ? "Dirichlet" : "Neumann");
    printf("Debug Mode: %s\n", debug_mode ? "Enabled" : "Disabled");
    if (debug_mode) {
        printf("  Max Time Steps: %d\n", MAX_TIME_STEPS);
        printf("  Heat Chance: %d%%\n", PERCENT_ADD_HEAT_CHANCE);
    }
    printf("====================================\n");

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
                add_heat_launcher(x, y);
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
    h_u0.clear();
    h_u1.clear();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}