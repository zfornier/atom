#ifndef __PLASMA_NETCDF__
#define __PLASMA_NETCDF__

#include <fstream>
#include <string>
#include <iostream>
#include <cstdlib>
#include <netcdf>
#include <vector>
#include <unistd.h>


typedef int Error_t;

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

#define UNITS string("units")
#define DESCRIPTION string("description")

#define PLSM_OK                     0
#define PLSM_ERROR_FILE_NOT_FOUND   1

#define PLSM_NC_ERROR 2

namespace plasmanetcdf {
	class NetCDFManipulator {
	public :
	/*!
		Create an empty NetCDF file 
		\param fileName    	Name of the NetCDF file to create
		\param mesh_size   	Size of the mesh
	
	*/
	static Error_t plsm_create(const char* fileName, const int mesh_size[3]);

	/*!
		Save new dimensions in the NetCDF file
		\param fileName    	Name of the NetCDF file to open in writing mode
		\param dim_names    Names of the 3 dimensions 
		\param dim_sizes    Size of all the dimensions 
		\param dim_nb    	Number of dimensions (here 3) 
	*/
	static Error_t plsm_save_info(const char *fileName, const char* dim_names[], const int dim_sizes[], const int dim_nb);

	/*!
		Get the description of a variable in the NetCDF file
		\param fileName    	Name of the NetCDF file to open in reading mode
		\param variable	Name of the variable you want to get the description from 
	*/
	const char* plsm_get_description(const char *fileName, const char* variable);

	/*!
		Get the value of a variable in the NetCDF file 
		\param fileName    Name of the NetCDF file to open in reading mode
		\param name        Name of the variable to get
		\param array       A var of whatever type 
	 
	*/
	static Error_t plsm_get_var(const char* fileName, const char * name, void* array);

	/*!
		Get a list of all arrays names in the NetCDF file (Not working yet)
		\param fileName    Name of the NetCDF file to open in writing mode
		\param variable    Testing
		
	*/
	static Error_t plsm_get_arrays_names(const char *fileName, const char* variable);

	/*!
		Save a 3d array in the NetCDF file
		\param fileName    Name of the NetCDF file to open in writing mode
		\param pTab        Tab containing all the data of the 3d array, doubles  
		\param label       Name of the 3d array to save
		\param unit	       Units of the 3d array
		\param desc	       Description of the 3d array 
	
	*/

	static Error_t plsm_save_3D_double_array(const char* fileName, double* pTab, char* label,const char* unit,const char* desc) ;

	/*!
		Save a 1d array in the NetCDF file
		\param fileName    Name of the NetCDF file to open in writing mode
		\param pTab        Tab containing all the data in  of the 1d array, ints
		\param label       Name of the 1d array to save
		\param unit	       Units of the 1d array
		\param desc	       Description of the 1d array   
	*/
	static Error_t plsm_save_1D_int_array(const char* fileName, int* pTab, char* label, char* dim_label,const char* unit,const char* desc);

	/*!
		Save a double variable in the NetCDF file
		\param fileName    Name of the NetCDF file to open in writing mode
		\param pTab        Tab containing the double to save
		\param label       Name of the double
		\param unit	       Units of the value
		\param desc	       Description of this value
	*/
	static Error_t plsm_save_double(const char* fileName, double* pTab, char* label,const char* unit,const char* desc);

	/*!
		Save a integer variable in the NetCDF file
		\param fileName    Name of the NetCDF file to open in writing mode
		\param pTab        Tab containing the int to save
		\param label       Name of the int
		\param unit	       Units of the value
		\param desc	       Description of this value
	*/
	static Error_t plsm_save_int(const char* fileName, int* pTab, char* label, const char* unit,const char* desc);


	/*!
		Add an attribute to a given variable
		\param fileName    	Name of the NetCDF file to open in writing mode
		\param variable	Name of the variable to which you want to add an attribute
		\param name 	Name of the new attribute
		\param desc		Description of the attribute 	
	
	*/
	static Error_t plsm_save_attribute(const char *fileName, const char* variable,const char* name,const char* desc);

	/*!
		Add a dimension with its size in the NetCDF File
	*/
	static Error_t plsm_add_dimension(const char *fname, const char* dim_name, int dim_size);

	};
}
#endif



