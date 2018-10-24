cmake_minimum_required(3.8)
project(atom)

find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)

set(CUDAFLAGS ${CUDAFLAGS}; -arch=sm_52 -lineinfo --maxrregcount=128 -g -I$(CUDA)/include/)
set(LDFLAGS ${LDFLAGS}; -lm -L$(CUDALIB) -arch=sm_52)