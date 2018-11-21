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


if(NOT ${CMAKE_BUILD_TYPE} STREQUAL "")
    string( TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER )

    if( ( NOT ${CMAKE_BUILD_TYPE_LOWER} STREQUAL "none" ) AND
        ( NOT ${CMAKE_BUILD_TYPE_LOWER} STREQUAL "release" ) AND
        ( NOT ${CMAKE_BUILD_TYPE_LOWER} STREQUAL "debug" ) AND
        ( NOT ${CMAKE_BUILD_TYPE_LOWER} STREQUAL "relwithdebinfo" ) AND
        ( NOT ${CMAKE_BUILD_TYPE_LOWER} STREQUAL "profiling" ))
         message( FATAL_ERROR "CMAKE_BUILD_TYPE : '${CMAKE_BUILD_TYPE}' is not a valid build type. "
        "Valid options are: 'None', 'Release', 'Debug', 'RelWithDebInfo', and 'Profiling'." )
    endif()

    message(STATUS "BUILD TYPE set to '${CMAKE_BUILD_TYPE}'")
    SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of
build, options are: 'None', 'Release', 'Debug', 'RelWithDebInfo', and 'Profiling'.")
else()
    SET(CMAKE_BUILD_TYPE "Release")
    message(STATUS "BUILD TYPE set to the default type:  '${CMAKE_BUILD_TYPE}'")
    string( TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER )
endif()

# ------------------OPTIONS WHICH ARE NEEDED TO SET BUILD-TYPES (COMPILATION FLAGS)--
# ------------------------------------------------------------------------CUDA-ARCH--
if(NOT DEFINED CUDA_ARCH)
    message(STATUS "Setting fallback CUDA_ARCH=35")
    set(CUDARCH "-arch=sm_35")
else(NOT DEFINED CUDA_ARCH)
    message(STATUS "Using provided CUDA_ARCH=${CUDA_ARCH}")
    set(CUDARCH "-arch=sm_${CUDA_ARCH}")
endif(NOT DEFINED CUDA_ARCH)


SET(NVCC_OPTS "")
set(CXX_OPTS "")
if(${CMAKE_BUILD_TYPE_LOWER} STREQUAL "debug")
    set(NVCC_OPTS "-lineinfo -O0 ${NVCC_OPTS}")
    set(CXX_OPTS "-g -O0 ${CXX_OPTS}")
endif()

if(${CMAKE_BUILD_TYPE_LOWER} STREQUAL "release")
    set(NVCC_OPTS "-O3 ${NVCC_OPTS}")
    set(CXX_OPTS "-O3 ${CXX_OPTS}")
endif()

if(${CMAKE_BUILD_TYPE_LOWER} STREQUAL "relwithdebinfo")
    set(NVCC_OPTS "-O3 -g ${NVCC_OPTS}")
    set(CXX_OPTS "-O3 -lineinfo ${CXX_OPTS}")
endif()

if(${CMAKE_BUILD_TYPE_LOWER} STREQUAL "profiling")
    set(NVCC_OPTS "-O3 -DTIMING ${NVCC_OPTS}")
    set(CXX_OPTS "-O3 -DTIMING ${CXX_OPTS}")
endif()

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 ${CXX_OPTS} -Wall")

LIST(APPEND CMAKE_CUDA_FLAGS "${NVCC_OPTS} ${CUDARCH} --maxrregcount=128 ")


ADD_SUBDIRECTORY(src/NetCdf)

ADD_EXECUTABLE(atom ${SOURCES})
TARGET_LINK_LIBRARIES(atom ${CUDA_LIBRARIES} ${MPI_LIBRARIES} netcdf_lib ${NETCDF_CXX_LIBRARIES_C})

