#include "../../../include/NetCdf/read_file.h"

int copyFile(const char *binaryFileName, const char *netCdfFileName) {
    ifstream ifs(binaryFileName, ios::binary);
    NcFile dataFile(netCdfFileName, NcFile::replace);
    dataFile.close();
    int res;

    // copy dimensions
    NetCDFManipulator::plsm_add_dimension(netCdfFileName, "x", NX);
    NetCDFManipulator::plsm_add_dimension(netCdfFileName, "y", NY);
    NetCDFManipulator::plsm_add_dimension(netCdfFileName, "z", NZ);

/*   	NcDim xDim = dataFile.addDim("x", NX);	
	NcDim yDim = dataFile.addDim("y", NY);
	NcDim zDim = dataFile.addDim("z", NZ);
  */
    copyOne3DArray(ifs, netCdfFileName, ELECTRIC_FIELD_LABEL + X_LABEL, UNITS_ELECTRIC_FIELD,
                   DESC_ELECTRIC_FIELD + ELECTRIC_FIELD_LABEL + X_LABEL);
    copyOne3DArray(ifs, netCdfFileName, ELECTRIC_FIELD_LABEL + Y_LABEL, UNITS_ELECTRIC_FIELD,
                   DESC_ELECTRIC_FIELD + ELECTRIC_FIELD_LABEL + Y_LABEL);
    copyOne3DArray(ifs, netCdfFileName, ELECTRIC_FIELD_LABEL + Z_LABEL, UNITS_ELECTRIC_FIELD,
                   DESC_ELECTRIC_FIELD + ELECTRIC_FIELD_LABEL + Z_LABEL);

    copyOne3DArray(ifs, netCdfFileName, MAGNETIC_FIELD_LABEL + X_LABEL, UNITS_MAGNETIC_FIELD,
                   DESC_MAGNETIC_FIELD + MAGNETIC_FIELD_LABEL + X_LABEL);
    copyOne3DArray(ifs, netCdfFileName, MAGNETIC_FIELD_LABEL + Y_LABEL, UNITS_MAGNETIC_FIELD,
                   DESC_MAGNETIC_FIELD + MAGNETIC_FIELD_LABEL + Y_LABEL);
    copyOne3DArray(ifs, netCdfFileName, MAGNETIC_FIELD_LABEL + Z_LABEL, UNITS_MAGNETIC_FIELD,
                   DESC_MAGNETIC_FIELD + MAGNETIC_FIELD_LABEL + Z_LABEL);

    copyOne3DArray(ifs, netCdfFileName, CURRENT_FIELD_LABEL + X_LABEL, UNITS_NO,
                   CURRENT + CURRENT_FIELD_LABEL + X_LABEL);
    copyOne3DArray(ifs, netCdfFileName, CURRENT_FIELD_LABEL + Y_LABEL, UNITS_NO,
                   CURRENT + CURRENT_FIELD_LABEL + Y_LABEL);
    copyOne3DArray(ifs, netCdfFileName, CURRENT_FIELD_LABEL + Z_LABEL, UNITS_NO,
                   CURRENT + CURRENT_FIELD_LABEL + Z_LABEL);

    copyOne3DArray(ifs, netCdfFileName, MAGNETIC_HALF_STEP_FIELD_LABEL + X_LABEL, UNITS_NO,
                   DESC_HALFSTEP + MAGNETIC_HALF_STEP_FIELD_LABEL + X_LABEL);
    copyOne3DArray(ifs, netCdfFileName, MAGNETIC_HALF_STEP_FIELD_LABEL + Y_LABEL, UNITS_NO,
                   DESC_HALFSTEP + MAGNETIC_HALF_STEP_FIELD_LABEL + Y_LABEL);
    copyOne3DArray(ifs, netCdfFileName, MAGNETIC_HALF_STEP_FIELD_LABEL + Z_LABEL, UNITS_NO,
                   DESC_HALFSTEP + MAGNETIC_HALF_STEP_FIELD_LABEL + Z_LABEL);


    copyOneSortParticle(ifs, netCdfFileName, SORT_0_LABEL);
    copyOneSortParticle(ifs, netCdfFileName, SORT_1_LABEL);
    copyOneSortParticle(ifs, netCdfFileName, SORT_2_LABEL);


    ifs.close();

    return 0;
}

int copyOneSortParticle(ifstream &ifs, const char *netCdfFileName, string label) {
    copyInt(ifs, netCdfFileName, EXTRA_NUMBER_LABEL + label, UNITS_NO, DESC_EXTRA);

    int nb_particles = (int) *readDouble(ifs);
    string nb_label = NB_PARTICLES_LABEL + label;
    char *pLabel = new char[nb_label.length() + 1];
    strcpy(pLabel, nb_label.c_str());
    string unit = UNITS_NB_PARTICLES;
    char *pUnit = new char[unit.length() + 1];
    strcpy(pUnit, unit.c_str());
    string desc = DESC_NB_PARTICLES + label;
    char *pDesc = new char[desc.length() + 1];
    strcpy(pDesc, desc.c_str());

    NetCDFManipulator::plsm_save_int(netCdfFileName, &nb_particles, pLabel, pUnit, pDesc);
    string name_dim = label + string("_DIM");
    char *pDim = new char[name_dim.length() + 1];
    strcpy(pDim, name_dim.c_str());
    NetCDFManipulator::plsm_add_dimension(netCdfFileName, pDim, nb_particles);

    copyDouble(ifs, netCdfFileName, CHARGE_LABEL + label, UNITS_CHARGE_PARTICLES, DESC_CHARGE + label);
    copyDouble(ifs, netCdfFileName, MASS_LABEL + label, UNITS_MASS_PARTICLES, DESC_MASS + label);
    //extra number
    readInt(ifs);

    copy1dArray(ifs, netCdfFileName, COORDINATES_LABEL + X_LABEL + label, name_dim, UNITS_NO,
                X_LABEL + DESC_COORDINATES + label);
    copy1dArray(ifs, netCdfFileName, COORDINATES_LABEL + Y_LABEL + label, name_dim, UNITS_NO,
                Y_LABEL + DESC_COORDINATES + label);
    copy1dArray(ifs, netCdfFileName, COORDINATES_LABEL + Z_LABEL + label, name_dim, UNITS_NO,
                Z_LABEL + DESC_COORDINATES + label);

    copy1dArray(ifs, netCdfFileName, IMPULSES_LABEL + X_LABEL + label, name_dim, UNITS_IMPULSES,
                X_LABEL + DESC_IMPULSES + label);
    copy1dArray(ifs, netCdfFileName, IMPULSES_LABEL + Y_LABEL + label, name_dim, UNITS_IMPULSES,
                Y_LABEL + DESC_IMPULSES + label);
    copy1dArray(ifs, netCdfFileName, IMPULSES_LABEL + Z_LABEL + label, name_dim, UNITS_IMPULSES,
                Z_LABEL + DESC_IMPULSES + label);
    return 0;
}

int copyOne3DArray(ifstream &ifs, const char *filename, string label, string unit, string desc) {

    double *tdArray;
    int res;
    tdArray = read3dArray(ifs);

    char *pLabel = new char[label.length() + 1];
    strcpy(pLabel, label.c_str());
    char *pUnit = new char[unit.length() + 1];
    strcpy(pUnit, unit.c_str());
    char *pDesc = new char[desc.length() + 1];
    strcpy(pDesc, desc.c_str());
    res = NetCDFManipulator::plsm_save_3D_double_array(filename, tdArray, pLabel, pUnit, pDesc);
    delete[] pLabel;
    delete[] pUnit;
    delete[] pDesc;
    free(tdArray);

    return res;
}

double *read3dArray(ifstream &ifs) {

    // read length of array
    char *pBufferLength = (char *) malloc(sizeof(int));

    int *pLength = (int *) pBufferLength;
    ifs.read(pBufferLength, sizeof(int));
    pLength = (int *) pBufferLength;

    // read content of the array (there are pLength doubles)
    char *pBuffer = (char *) malloc((sizeof(double) * (*pLength)));
    ifs.read(pBuffer, (*pLength));

    // read length of array again (end of the array)
    ifs.read(pBufferLength, sizeof(int));

    double *pTab = (double *) pBuffer;

    free(pBufferLength);


    return pTab;
}

int copy1dArray(ifstream &ifs, const char *netCdfFileName, string label, string dim_label, string unit, string desc) {
    int *tdArray;
    int res;
    tdArray = read1dArray(ifs);
    char *pLabel = new char[label.length() + 1];
    strcpy(pLabel, label.c_str());
    char *pUnit = new char[unit.length() + 1];
    strcpy(pUnit, unit.c_str());
    char *pDesc = new char[desc.length() + 1];
    strcpy(pDesc, desc.c_str());
    char *pDim = new char[dim_label.length() + 1];
    strcpy(pDim, dim_label.c_str());
    res = NetCDFManipulator::plsm_save_1D_int_array(netCdfFileName, tdArray, pLabel, pDim, pUnit, pDesc);
    delete[] pLabel;
    delete[] pUnit;
    delete[] pDesc;
    delete[] pDim;
    free(tdArray);

    return res;
}

int *read1dArray(ifstream &ifs) {

    // read length of array
    char *pBufferLength = (char *) malloc(sizeof(int));

    int *pLength = (int *) pBufferLength;
    ifs.read(pBufferLength, sizeof(int));
    pLength = (int *) pBufferLength;

    // read content of the array (there are pLength ints)
    char *pBuffer = (char *) malloc((sizeof(int) * (*pLength)));
    ifs.read(pBuffer, (*pLength));

    // read length of array again (end of the array)
    ifs.read(pBufferLength, sizeof(int));

    int *pTab = (int *) pBuffer;

    free(pBufferLength);

    return pTab;
}

int copyDouble(ifstream &ifs, const char *netCdfFileName, string label, string unit, string desc) {
    double *tdArray;
    int res;
    tdArray = readDouble(ifs);
    char *pLabel = new char[label.length() + 1];
    strcpy(pLabel, label.c_str());
    char *pUnit = new char[unit.length() + 1];
    strcpy(pUnit, unit.c_str());
    char *pDesc = new char[desc.length() + 1];
    strcpy(pDesc, desc.c_str());
    res = NetCDFManipulator::plsm_save_double(netCdfFileName, tdArray, pLabel, pUnit, pDesc);
    delete[] pLabel;
    free(tdArray);

    return res;
}

double *readDouble(ifstream &ifs) {

    char *pBuffer = (char *) malloc(sizeof(double));
    ifs.read(pBuffer, sizeof(double));

    double *pTab = (double *) pBuffer;

    return pTab;
}

int copyInt(ifstream &ifs, const char *netCdfFileName, string label, string unit, string desc) {
    int *tdArray;
    int res;
    tdArray = readInt(ifs);
    char *pLabel = new char[label.length() + 1];
    strcpy(pLabel, label.c_str());
    char *pUnit = new char[unit.length() + 1];
    strcpy(pUnit, unit.c_str());
    char *pDesc = new char[desc.length() + 1];
    strcpy(pDesc, desc.c_str());
    res = NetCDFManipulator::plsm_save_int(netCdfFileName, tdArray, pLabel, pUnit, pDesc);
    delete[] pLabel;
    delete[] pUnit;
    delete[] pDesc;

    free(tdArray);

    return res;
}

int *readInt(ifstream &ifs) {

    char *pBuffer = (char *) malloc(sizeof(int));
    ifs.read(pBuffer, sizeof(int));

    int *pTab = (int *) pBuffer;

    return pTab;
}


void readVar(const char *fileName, const char *name, void *array) {
    NetCDFManipulator::plsm_get_var(fileName, name, array);
}
