# Compiler and flags
CXX := g++
CXX_FLAGS := -O3 -Wall -std=c++11

# Required libraries
LIB := -lglfw -lGL -lGLEW -lm -lGLU -lglut

# Project files
EXECUTABLE := heat_sim
SRC_FILES := heat_sim.cpp
OBJ_FILES := $(SRC_FILES:.cpp=.o)

# Default target
all: $(EXECUTABLE)

# Link the executable
$(EXECUTABLE): $(OBJ_FILES)
	$(CXX) $(CXX_FLAGS) -o $@ $(OBJ_FILES) $(LIB)

# Compile each .cpp file to .o
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJ_FILES) $(EXECUTABLE)

# Phony targets
.PHONY: all clean
