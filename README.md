# TUDA-PMPP-CudaMemGather
Gathering data for CUDA memory visualization

# Build instructions

1. Build the modified mem_trace library by running `make` in `nvbit/tools/mem_trace`
2. Build the binaries of exercise 1 by running `cmake -B build && cd build && make` in `nvbit/test-apps/ex1`

# Run instructions

Run the exercise code with `NVOUT=[path/to/output.csv] LD_PRELOAD=[path/to/mem_trace.so] [path/to/ex1] [ex1 parameters]` in order to generate a dump of all memory accesses.

# Credits
This software contains source code provided by NVIDIA Corporation. The NVBit framework in its entirety can be found [here](https://github.com/NVlabs/NVBit).
Apart from the core framework, the project based its functionality off the `mem_trace` example provided by NVLabs.