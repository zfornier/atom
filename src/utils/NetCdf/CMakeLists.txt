CMAKE_MINIMUM_REQUIRED(VERSION 3.8) 
PROJECT(netcdf_lib VERSION 0.1
	DESCRIPTION "Plasma modeling code/netcdf library"
	LANGUAGES CXX)

find_path (NETCDF_CXX_INCLUDES_C netcdf PATHS "$ENV{NETCDF_CXX_DIR}")
find_library (NETCDF_CXX_LIBRARIES_C  netcdf-cxx4 PATHS "$ENV{NETCDF_CXX_LIB_DIR}")

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

set(SOURCES plasma_netcdf.cpp read_file.cpp write_file.cpp)

ADD_LIBRARY(netcdf_lib ${SOURCES})

