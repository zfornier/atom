#ifndef __READ_FILE__
#define __READ_FILE__

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <netcdf>
#include <vector>
#include <unistd.h>
#include "plasma_netcdf.h"

#define BINARY_FILE_NAME "mumu00000000001.dat"
#define NETCDF_FILE_NAME "file_out.nc"

#define ELECTRIC_FIELD_LABEL string("E")
#define MAGNETIC_FIELD_LABEL string("M")
#define CURRENT_FIELD_LABEL string("J")
#define MAGNETIC_HALF_STEP_FIELD_LABEL string("Q")

#define X_LABEL string("x")
#define Y_LABEL string("y")
#define Z_LABEL string("z")

#define EXTRA_NUMBER_LABEL string("Extra_number_")
#define MASS_LABEL string("Mass_")
#define CHARGE_LABEL string("Charge_")
#define NB_PARTICLES_LABEL string("Nb_particles_")
#define COORDINATES_LABEL string("Coordinates_")
#define IMPULSES_LABEL string("Impulses_")

#define SORT_0_LABEL string("0")
#define SORT_1_LABEL string("1")
#define SORT_2_LABEL string("2")

using namespace std;

using namespace netCDF;
using namespace plasmanetcdf;
using namespace netCDF::exceptions;

static const int NX = 102;
static const int NY = 6;
static const int NZ = 6;
// Return this in event of a problem.
static const int NC_ERR = 2;

//Name of the units
#define UNITS string("units")
#define UNITS_ELECTRIC_FIELD string("N.C^-1")
#define UNITS_MAGNETIC_FIELD string("T")
#define UNITS_NB_PARTICLES string("N^-1")

#define UNITS_CHARGE_PARTICLES string("C")
#define UNITS_MASS_PARTICLES  string(" ")
#define UNITS_NO string("no units ")
#define UNITS_IMPULSES string("N.s")

//Descriptions
#define DESCRIPTION  string("description")
#define DESC_ELECTRIC_FIELD string("electric field, ")
#define DESC_MAGNETIC_FIELD string("magnetic field, ")
#define CURRENT  string("current, ")
#define DESC_HALFSTEP string(" magnetic field at halfstep, ")
#define DESC_EXTRA  string("extra number placed by Fortran")
#define DESC_NB_PARTICLES  string("number of particles of sort ")
#define DESC_CHARGE string("charge for sort ")
#define DESC_MASS  string("mass for sort ")
#define DESC_COORDINATES  string(" coordinates for particles of sort ")
#define DESC_IMPULSES  string(" impulses for particles of sort ")

double *read3dArray(ifstream &ifs);

int copyOne3DArray(ifstream &ifs, const char *netCdfFileName, string label, string unit, string desc);

int copyInt(ifstream &ifs, const char *netCdfFileName, string label, string unit, string desc);

int *readInt(ifstream &ifs);

int copyDouble(ifstream &ifs, const char *netCdfFileName, string label, string unit, string desc);

double *readDouble(ifstream &ifs);

int copy1dArray(ifstream &ifs, const char *netCdfFileName, string label, string dim_label, string unit, string desc);

int *read1dArray(ifstream &ifs);

int copyOneSortParticle(ifstream &ifs, const char *netCdfFileName, string label);

int copyFile(const char *binaryFileName, const char *netCdfFileName);

void readVar(const char *fileName, const char *name, void *array);

int writeOne3DArray(const char *filename, double *tdArray, string label, string unit, string desc);

#endif

