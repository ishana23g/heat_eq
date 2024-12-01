# Compiler and flags
NVCC := nvcc
# complied on a 4060TI 16 GB card. So using the arch=sm_89
NVCC_FLAGS := --use_fast_math -O3 -gencode arch=compute_89,code=sm_89
# need to install the following libraries
# sudo apt-get install libglfw3-dev libglfw3 libglew-dev libglew2.1 libglu1-mesa-dev
# need to use -I/usr/include -ccbin g++, because the default nvcc compiler is not able to find the right files otherwise
LIB := -lcudart -lcurand -lcublas -lcufft  -I/usr/include -ccbin g++ -lglfw -lGL -lGLEW -lGLU -lglut

# # Project files
# EXECUTABLE := cuda_heat_equation

EXECUTABLE := cuda_heat_equation

SRC_FILES := $(EXECUTABLE).cu

OBJ_FILES := $(SRC_FILES:.cu=.o)

# Build the project
all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJ_FILES)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJ_FILES) $(LIB)

# Compile each .cu file to .o
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(LIB)

# Clean up build files
clean:
	rm -f $(OBJ_FILES) $(EXECUTABLE)