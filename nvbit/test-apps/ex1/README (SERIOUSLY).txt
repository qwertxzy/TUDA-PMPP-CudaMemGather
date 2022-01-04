Note: This will not compile on GCCG301 due to the cuda binaries not being on the PATH environment variable.
      It is still possible to compile, but you have to manually set an environment variable to the binary location
      of CUDA. I have written an e-mail to Prof. Guthe about this and hope this will be resolved by the time we have
      to hand in ex2.

Note: You NEED the CUDA Toolkit installed under /usr/local/cuda, otherwise this won't compile without changing the
      include_directories path inside CMakeLists.

Compile and Run:
    cmake -B build
    cd build
    make
    ./matmul random compare

Note that if you want to see the element comparison, it must be specified by passing the command line parameter 'compare'.
I disabled it by default because it takes a long time to print for larger matrices.