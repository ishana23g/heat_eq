# Makefile

# Compiler and flags
NVCC := nvcc
# Set the appropriate architecture for your GPU
NVCC_FLAGS := --use_fast_math -O3 -gencode arch=compute_86,code=sm_86 -Xptxas -v

# Required libraries
LIB := -lcudart -lcurand -lcublas -lcufft -I/usr/include -ccbin g++
LIB_GL := -lglfw -lGL -lGLEW -lGLU -lglut
# Project files
EXECUTABLE := heat_sim
CU_FILES := main.cu kernels.cu simulation.cu rendering.cu
HEADERS := kernels.h simulation.h rendering.h utility.h constants.h

OBJ_FILES := $(CU_FILES:.cu=.o) $(CPP_FILES:.cpp=.o)

# Default target
all: $(EXECUTABLE)

# Link the executable
$(EXECUTABLE): $(OBJ_FILES)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJ_FILES) $(LIB) $(LIB_GL)

# Compile .cu files to .o
%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(LIB) $(LIB_GL)

# Clean up build files
clean:
	rm -f $(OBJ_FILES) $(EXECUTABLE)