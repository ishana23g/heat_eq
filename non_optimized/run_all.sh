#!/bin/bash

# Make the CUDA program
make 

O=(-d -m 2d -b n)

# Run diagnostics of memory and cache
# check if diagnosis directory exists
if [ ! -d "diagnosis" ]; then
    mkdir diagnosis
fi
# clear the diagnosis directory
rm -rf diagnosis/*
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./cuda_heat_equation "${O[@]}" 2>&1 | tee diagnosis/memcheck.txt
valgrind --tool=cachegrind ./cuda_heat_equation "${O[@]}" 2>&1 | tee diagnosis/cachegrind.txt
compute-sanitizer ./cuda_heat_equation "${O[@]}"  2>&1 | tee diagnosis/sanitizer.txt

# Run the profiler for the CUDA program
# check if profiler directory exists
if [ ! -d "profiler" ]; then
    mkdir profiler
fi
# clear the profiler directory
rm -rf profiler/*
ncu ./cuda_heat_equation "${O[@]}" 2>&1 | tee profiler/ncu_all.txt
# measure how much FMAs and memory bandwidth is being used
ncu --metrics sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
sm__sass_thread_inst_executed_op_imul_pred_on.sum,\
sm__sass_thread_inst_executed_op_iadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_imad_pred_on.sum \
./cuda_heat_equation "${O[@]}" 2>&1 | tee profiler/ncu_fma_memory.txt

ncu --print-summary per-kernel ./cuda_heat_equation "${O[@]}" 2>&1 | tee profiler/ncu_summary.txt