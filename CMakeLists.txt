cmake_minimum_required(VERSION 3.8)
project(atom)

find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)

set(CUDAFLAGS "-arch=sm_52;-lineinfo;--maxrregcount=128;-g;-I${CUDA_TOOLKIT_ROOT_DIR}/include/")
set(CUDA_LIB "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
set(CUDA_LIBS "-g;-L${CUDA_LIB};-lcuda;-lcudart")
set(OBJ  "main.o;mpi_shortcut.o;service_functions.o;compare.o;maxwell.o;load_data.o;archAPI.o")
set(MPI_COMP "mpicxx")
set(CUDA_COMP "nvcc")
set(CPP "g++")
set(NET_CDF_LIB "../netcdf-cxx4/cxx4")
set(NET_CDF_BLD "../netcdf-cxx4/build/cxx4/")
set(NET_CDF_FLAGS "-lnetcdf-cxx4;-lnetcdf;-lhdf5_serial_hl;-lhdf5_serial;-lz")

add_custom_target(plasma_netcdf.o ALL)
add_custom_command(TARGET plasma_netcdf.o COMMAND ${CPP} -c -I${NET_CDF_LIB} src/NetCdf/plasma_netcdf.cpp)

add_custom_target(libplasmanetcdf.a ALL)
add_custom_command(TARGET libplasmanetcdf.a COMMAND ar rcs libplasmanetcdf.a plasma_netcdf.o)
add_dependencies(libplasmanetcdf.a plasma_netcdf.o)

add_custom_target(main.o ALL)
add_custom_command(TARGET main.o COMMAND ${CUDA_COMP} -g -c src/main.cu ${CUDAFLAGS} -I${NET_CDF_LIB})

add_custom_target(mpi_shortcut.o ALL)
add_custom_command(TARGET mpi_shortcut.o COMMAND ${MPI_COMP} -g -c src/mpi_shortcut.cxx)

add_custom_target(service_functions.o ALL)
add_custom_command(TARGET service_functions.o COMMAND ${CUDA_COMP} -g -c src/service_functions.cu ${CUDAFLAGS})

add_custom_target(compare.o ALL)
add_custom_command(TARGET compare.o COMMAND ${MPI_COMP} -g -c src/compare.cxx)

add_custom_target(maxwell.o ALL)
add_custom_command(TARGET maxwell.o COMMAND ${CUDA_COMP} -g -c src/maxwell.cu ${CUDAFLAGS})

add_custom_target(load_data.o ALL)
add_custom_command(TARGET load_data.o COMMAND ${CUDA_COMP} -std=c++11 -g -c src/load_data.cu ${CUDAFLAGS} -I${NET_CDF_LIB} -L${NET_CDF_BLD} libplasmanetcdf.a ${NET_CDF_FLAGS})

add_dependencies(load_data.o libplasmanetcdf.a)

add_custom_target(archAPI.o ALL)
add_custom_command(TARGET archAPI.o COMMAND ${CUDA_COMP}  -g -c src/archAPI.cu ${CUDAFLAGS})

add_custom_target(atom ALL)
add_custom_command(TARGET atom COMMAND ${MPI_COMP} -g -o atom ${OBJ} -L${NET_CDF_BLD} libplasmanetcdf.a ${NET_CDF_FLAGS} ${CUDA_LIBS})

add_dependencies(atom main.o)
add_dependencies(atom mpi_shortcut.o)
add_dependencies(atom service_functions.o)
add_dependencies(atom compare.o)
add_dependencies(atom maxwell.o)
add_dependencies(atom load_data.o)
add_dependencies(atom archAPI.o)
add_dependencies(atom libplasmanetcdf.a)
