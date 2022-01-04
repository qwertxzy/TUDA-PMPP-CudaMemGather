#/bin/bash

#echo "BUILDING VECTORADD..."
#cd ~/project/nvbit/test-apps/
#make

echo "BUILDING MEMTRACE..."
cd ~/project/nvbit/tools/mem_trace
make

echo "EXECUTING..."
rm ~/project/out.csv
# NVOUT=~/project/out.csv LD_PRELOAD=~/project/nvbit/tools/mem_trace/mem_trace.so ~/project/nvbit/test-apps/vectoradd/vectoradd

NVOUT=~/project/out.csv LD_PRELOAD=~/project/nvbit/tools/mem_trace/mem_trace.so ~/project/nvbit/test-apps/ex1/build/matmul


# gdb -iex "set exec-wrapper env NVOUT=~/project/out.csv LD_PRELOAD=~/project/nvbit/tools/mem_trace/mem_trace.so" ~/project/nvbit/test-apps/vectoradd/vectoradd