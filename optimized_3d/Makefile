# Compiler and flags
NVCC := nvcc
# Compiled on a 4060TI 16 GB card. Using arch=sm_89
# Compiled on a 3060TI Mobile 8GB card. Using arch=sm_86
NVCC_FLAGS := --use_fast_math -O3 -gencode arch=compute_86,code=sm_86 -Xptxas -v
#,-abi=no

# Required libraries
# sudo apt-get install libglfw3-dev libglfw3 libglew-dev libglew2.1 libglu1-mesa-dev
# need to use -I/usr/include -ccbin g++, because the default nvcc compiler is not able to find the right files otherwise
LIB := -lcudart -lcurand -lcublas -lcufft -I/usr/include -ccbin g++ -lglfw -lGL -lGLEW -lGLU -lglut

# Project files
EXECUTABLE := cuda_heat_equation
SRC_FILES := $(EXECUTABLE).cu
OBJ_FILES := $(SRC_FILES:.cu=.o)
PTX_FILES := $(SRC_FILES:.cu=.ptx)
CUBIN_FILES := $(SRC_FILES:.cu=.cubin)

# Default target
all: $(EXECUTABLE)

# Link the executable
$(EXECUTABLE): $(OBJ_FILES)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJ_FILES) $(LIB)

# Compile each .cu file to .o
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(LIB)

# Generate PTX files
%.ptx: %.cu
	$(NVCC) $(NVCC_FLAGS) -ptx $< -o $@
	$(NVCC) $(NVCC_FLAGS) $<

# Generate CUBIN files
%.cubin: %.cu
	$(NVCC) $(NVCC_FLAGS) -cubin $< -o $@
	cuobjdump --dump-sass $@ > $(<:.cu=.sass)

# Clean up build files
clean:
	rm -f $(OBJ_FILES) $(EXECUTABLE) $(PTX_FILES) $(CUBIN_FILES)

# Target to build PTX and CUBIN files
ptx: $(PTX_FILES)
cubin: $(CUBIN_FILES)