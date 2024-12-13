#!/bin/bash

# Make the CUDA program
make 

P=./cuda_heat_equation
O3=(-d -b N)
O2=(-d -b N -m 2d)

# Run diagnostics of memory and cache
# check if diagnosis directory exists
if [ ! -d "diagnosis" ]; then
    mkdir diagnosis
fi
# clear the diagnosis directory
rm -rf diagnosis/*

## VALGRIND
# valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all $P "${O3[@]}" 2>&1 | tee diagnosis/memcheck.txt
# valgrind --tool=cachegrind $P "${O3[@]}" 2>&1 | tee diagnosis/cachegrind.txt

## CUDA-MEMCHECK - COMPUTE SANITIZER (NEW)
compute-sanitizer $P "${O3[@]}"  2>&1 | tee diagnosis/sanitizer_3D.txt
compute-sanitizer $P "${O2[@]}"  2>&1 | tee diagnosis/sanitizer_2D.txt

# Run the profiler for the CUDA program
# check if profiler directory exists
if [ ! -d "profiler" ]; then
    mkdir profiler
fi
# clear the profiler directory
rm -rf profiler/*

## NVIDIA VISUAL PROFILER
ncu $P "${O3[@]}" 2>&1 | tee profiler/ncu_all_3D.txt
ncu $P "${O2[@]}" 2>&1 | tee profiler/ncu_all_2D.txt
# measure how much FMAs and memory bandwidth is being used
ncu --metrics sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
$P "${O3[@]}" 2>&1 | tee profiler/ncu_fma_memory_3D.txt
ncu --metrics sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
$P "${O2[@]}" 2>&1 | tee profiler/ncu_fma_memory_2D.txt
ncu --print-summary per-kernel $P "${O3[@]}" 2>&1 | tee profiler/ncu_summary_3D.txt
ncu --print-summary per-kernel $P "${O2[@]}" 2>&1 | tee profiler/ncu_summary_2D.txt


## NVIDIA PROFILER - GUI
echo "==================================================================="
echo "For the NVIDIA Profiler GUI, run the following command:"
echo "nsys-ui"
echo "This allows you to view a timeline of of kernals, openGL calls, and memory transfers."
echo "Note that you will need to know where you are, (pwd) and what the program is called."
echo pwd 
echo $P "${O3[@]}"
echo 
echo "For the NVIDIA Profiler CLI, run the following command:"
echo ncu --open-in-ui -o profiler/ncu_report $P "${O3[@]}"
echo "If that does not work" 
echo "Just run `ncu` and open the directory manually, and specify the program and options."