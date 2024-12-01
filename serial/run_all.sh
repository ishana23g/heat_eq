#!/bin/bash

# Make the program
make 

O=(-d)

# Run diagnostics of memory and cache
# check if diagnosis directory exists
if [ ! -d "diagnosis" ]; then
    mkdir diagnosis
fi
# clear the diagnosis directory
rm -rf diagnosis/*
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./heat_sim "${O[@]}" 2>&1 | tee diagnosis/memcheck.txt
valgrind --tool=cachegrind ./heat_sim "${O[@]}" 2>&1 | tee diagnosis/cachegrind.txt

# Run the profiler for the program
# check if profiler directory exists
if [ ! -d "profiler" ]; then
    mkdir profiler
fi
# clear the profiler directory
rm -rf profiler/*
perf stat -d ./heat_sim "${O[@]}" 2>&1 | tee profiler/perf_stat.txt

# Measure wall clock time
echo "Measuring wall clock time..."
(time ./heat_sim "${O[@]}") 2>&1 | tee profiler/wall_clock_time.txt
