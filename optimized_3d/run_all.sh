#!/bin/bash

# Make the CUDA program
make 

O=(-d -b N)

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
ncu --open-in-ui -o profiler/ncu_report ./cuda_heat_equation "${O[@]}" 
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

# Generate PTX files
echo "Generating PTX files..."
for cu_file in *.cu; do
    ptx_file="${cu_file%.cu}.ptx"
    $(NVCC) $(NVCC_FLAGS) $(PTX_FLAGS) -ptx $cu_file -o $ptx_file
done

# Generate CUBIN files
echo "Generating CUBIN files..."
for cu_file in *.cu; do
    cubin_file="${cu_file%.cu}.cubin"
    sass_file="${cu_file%.cu}.sass"
    $(NVCC) $(NVCC_FLAGS) -cubin $cu_file -o $cubin_file
    cuobjdump --dump-sass $cubin_file > $sass_file
done

# Measure wall clock time
echo "Measuring wall clock time..."
(time ./cuda_heat_equation "${O[@]}") 2>&1 | tee profiler/wall_clock_time.txt