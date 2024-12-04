#!/bin/bash

# Make the CUDA program
make 

P=./cuda_heat_equation
O=(-d -b N)

# Run diagnostics of memory and cache
# check if diagnosis directory exists
if [ ! -d "diagnosis" ]; then
    mkdir diagnosis
fi
# clear the diagnosis directory
rm -rf diagnosis/*

## VALGRIND
# valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all $P "${O[@]}" 2>&1 | tee diagnosis/memcheck.txt
# valgrind --tool=cachegrind $P "${O[@]}" 2>&1 | tee diagnosis/cachegrind.txt

## CUDA-MEMCHECK - COMPUTE SANITIZER (NEW)
compute-sanitizer $P "${O[@]}"  2>&1 | tee diagnosis/sanitizer.txt

# Run the profiler for the CUDA program
# check if profiler directory exists
if [ ! -d "profiler" ]; then
    mkdir profiler
fi
# clear the profiler directory
rm -rf profiler/*

## NVIDIA VISUAL PROFILER
ncu $P "${O[@]}" 2>&1 | tee profiler/ncu_all.txt
# measure how much FMAs and memory bandwidth is being used
ncu --metrics sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
$P "${O[@]}" 2>&1 | tee profiler/ncu_fma_memory.txt
ncu --print-summary per-kernel $P "${O[@]}" 2>&1 | tee profiler/ncu_summary.txt

## NVIDIA PROFILER - GUI
echo "For the NVIDIA Profiler GUI, run the following command:"
echo "nsys-ui"
echo "This allows you to view a timeline of of kernals, openGL calls, and memory transfers."
echo "Note that you will need to know where you are, (pwd) and what the program is called."
echo pwd 
echo $P "${O[@]}"
echo 
echo "For the NVIDIA Profiler CLI, run the following command:"
echo ncu --open-in-ui -o profiler/ncu_report $P "${O[@]}"