# Compiler and flags
NVCC := nvcc
# complied on a 4060TI 16 GB card. So using the arch=sm_89
NVCC_FLAGS := --use_fast_math -O3 -arch=sm_89
# need to install the following libraries
# sudo apt-get install libglfw3-dev libglfw3 libglew-dev libglew2.1
# need to use -I/usr/include -ccbin g++, because the default nvcc compiler is not able to find the right files otherwise
LIB := -lcudart -lcurand -lcublas -lcufft  -I/usr/include -ccbin g++ -lglfw -lGL -lGLEW 

# Project files
EXECUTABLE := cuda_heat_equation

SRC_FILES := cuda_heat_equation.cu


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

make all:
	nvcc -o cuda_heat_equation cuda_heat_equation.cu -I/usr/include -ccbin g++ -lglfw -lGL -lGLEW
