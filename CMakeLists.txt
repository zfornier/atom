cmake_minimum_required(VERSION 3.8)
project(atom)

find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)

#set(CUDAFLAGS ${CUDAFLAGS}; -arch=sm_52 -lineinfo --maxrregcount=128 -g -I$(CUDA)/include/)
#set(LDFLAGS ${LDFLAGS}; -lm -L$(CUDALIB) -arch=sm_52)
#set(OBJ  "main.o" "mpi_shortcut.o" "service_functions.o" "compare.o" "maxwell.o" "load_data.o" "archAPI.o")

add_custom_target(main.o ALL)
add_custom_command(TARGET main.o COMMAND nvcc -g -c src/main.cu -arch=sm_52 -lineinfo --maxrregcount=128 -g -I/usr/local/cuda/include/ -I../netcdf-cxx4/cxx4/)

add_custom_target(mpi_shortcut.o ALL)
add_custom_command(TARGET mpi_shortcut.o COMMAND mpicxx -g -c  src/mpi_shortcut.cxx)

add_custom_target(service_functions.o ALL)
add_custom_command(TARGET service_functions.o COMMAND nvcc -g -c src/service_functions.cu -arch=sm_52 -lineinfo --maxrregcount=128 -g -I/usr/local/cuda/include/ )

add_custom_target(compare.o ALL)
add_custom_command(TARGET compare.o COMMAND mpicxx -g -c src/compare.cxx)

add_custom_target(maxwell.o ALL)
add_custom_command(TARGET maxwell.o COMMAND nvcc -g -c src/maxwell.cu -arch=sm_52 -lineinfo --maxrregcount=128 -g -I/usr/local/cuda/include/)

add_custom_target(load_data.o ALL)
add_custom_command(TARGET load_data.o COMMAND nvcc -g -c src/load_data.cu -arch=sm_52 -lineinfo --maxrregcount=128 -g -I/usr/local/cuda/include/)

add_custom_target(archAPI.o ALL)
add_custom_command(TARGET archAPI.o COMMAND nvcc -g -c src/archAPI.cu -arch=sm_52 -lineinfo --maxrregcount=128 -g -I/usr/local/cuda/include/)

add_custom_target(plasma_netcdf.o ALL)
add_custom_command(TARGET plasma_netcdf.o COMMAND g++ -I../netcdf-cxx4/cxx4 -c src/NetCdf/plasma_netcdf.cpp)

add_custom_target(libplasmanetcdf.a ALL)
add_custom_command(TARGET libplasmanetcdf.a COMMAND ar rcs libplasmanetcdf.a plasma_netcdf.o)
add_dependencies(libplasmanetcdf.a plasma_netcdf.o)

add_custom_target(atom ALL)
add_custom_command(TARGET atom
        COMMAND mpicxx -g -o atom main.o mpi_shortcut.o service_functions.o compare.o maxwell.o load_data.o archAPI.o -L../netcdf-cxx4/build/cxx4/ libplasmanetcdf.a -lnetcdf-cxx4 -lnetcdf -lhdf5_serial_hl -lhdf5_serial -lz -g -L/usr/local/cuda/lib64 -lcuda -lcudart)
add_dependencies(atom main.o)
add_dependencies(atom mpi_shortcut.o)
add_dependencies(atom service_functions.o)
add_dependencies(atom compare.o)
add_dependencies(atom maxwell.o)
add_dependencies(atom load_data.o)
add_dependencies(atom archAPI.o)
add_dependencies(atom libplasmanetcdf.a)