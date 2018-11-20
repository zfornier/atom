CMAKE_MINIMUM_REQUIRED(VERSION 3.8) 
PROJECT(atom VERSION 0.1
	DESCRIPTION "Plasma modeling code"
	LANGUAGES CXX CUDA)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(MPI REQUIRED)

find_path (NETCDF_CXX_INCLUDES_C netcdf PATHS "$ENV{NETCDF_CXX_DIR}")
find_library (NETCDF_CXX_LIBRARIES_C  netcdf-cxx4 PATHS "$ENV{NETCDF_CXX_LIB_DIR}")

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS} ${MPI_INCLUDE_PATH} ${NETCDF_CXX_INCLUDES_C})

set(SOURCES src/main.cu src/mpi_shortcut.cxx src/service_functions.cu src/compare.cxx src/maxwell.cu src/load_data.cu src/archAPI.cu)

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -ffast-math -Wall")

LIST(APPEND CUDA_NVCC_FLAGS "-lineinfo;-gencode;arch=compute_52,code=sm_52;--maxrregcount=128;-O3")

ADD_SUBDIRECTORY(src/NetCdf)

ADD_EXECUTABLE(atom ${SOURCES})
TARGET_LINK_LIBRARIES(atom ${CUDA_LIBRARIES} ${MPI_LIBRARIES} netcdf_lib ${NETCDF_CXX_LIBRARIES_C})

