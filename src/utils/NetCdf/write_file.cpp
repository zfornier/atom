//
// Created by egor on 26.02.19.
//

#include "../../../include/NetCdf/read_file.h"

int writeOne3DArray(const char *filename, double *tdArray, string label, string unit, string desc) {
    int res;

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

    return res;
}