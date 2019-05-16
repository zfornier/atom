/*
 * read_particle.cxx
 *
 *  Created on: Jun 9, 2018
 *      Author: snytav
 */

#include "../../include/load_data.h"

void readParticleParamsOneSort(const char * filename, int * total_particles, double *qq_m, double *mm, int sort) {
    double q_m, m;
    int total;

    //Reading total_particles for sort "sort"
    readVar(filename, (std::string("Nb_particles_") + patch::to_string(sort)).c_str(), &total);

    //Reading charge for sort "sort"
    readVar(filename, (std::string("Charge_") + patch::to_string(sort)).c_str(), &q_m);

    //Reading mass for sort "sort"
    readVar(filename, (std::string("Mass_") + patch::to_string(sort)).c_str(), &m);

    *qq_m = q_m;
    *mm = m;
    *total_particles = total;
}

void readBinaryParticleArraysOneSort(const char * filename, double *x, double *y, double *z, double *px, double *py, double *pz, int sort) {
    //Reading X coordinates for particles of sort "sort"
    readVar(filename, (std::string("Coordinates_x") + patch::to_string(sort)).c_str(), (void *) x);

    //Reading Y coordinates for particles of sort "sort"
    readVar(filename, (std::string("Coordinates_y") + patch::to_string(sort)).c_str(), (void *) y);

    //Reading Z coordinates for particles of sort "sort"
    readVar(filename, (std::string("Coordinates_z") + patch::to_string(sort)).c_str(), (void *) z);

    //Reading X impulses for particles of sort "sort"
    readVar(filename, (std::string("Impulses_x") + patch::to_string(sort)).c_str(), (void *) px);

    //Reading Y impulses for particles of sort "sort"
    readVar(filename, (std::string("Impulses_y") + patch::to_string(sort)).c_str(), (void *) py);

    //Reading Z impulses for particles of sort "sort"
    readVar(filename, (std::string("Impulses_z") + patch::to_string(sort)).c_str(), (void *) pz);
}

NetCdfData * getDataFromFile(const char * is) {
    NetCdfData * data = new NetCdfData;

    readDimVar(is, "x", &data->nx);
    readDimVar(is, "y", &data->ny);
    readDimVar(is, "z", &data->nz);

    data->Ex = new double[data->nx * data->ny * data->nz];
    data->Ey = new double[data->nx * data->ny * data->nz];
    data->Ez = new double[data->nx * data->ny * data->nz];

    data->Hx = new double[data->nx * data->ny * data->nz];
    data->Hy = new double[data->nx * data->ny * data->nz];
    data->Hz = new double[data->nx * data->ny * data->nz];

    data->Qy = new double[data->nx * data->ny * data->nz];
    data->Qx = new double[data->nx * data->ny * data->nz];
    data->Qz = new double[data->nx * data->ny * data->nz];

    data->Jx = new double[data->nx * data->ny * data->nz];
    data->Jy = new double[data->nx * data->ny * data->nz];
    data->Jz = new double[data->nx * data->ny * data->nz];

    readVar(is, "Ex", (void*)data->Ex);
    readVar(is, "Ey", (void*)data->Ey);
    readVar(is, "Ez", (void*)data->Ez);

    readVar(is, "Mx", (void*)data->Hx);
    readVar(is, "My", (void*)data->Hy);
    readVar(is, "Mz", (void*)data->Hz);

    readVar(is, "Qx", (void*)data->Qx);
    readVar(is, "Qy", (void*)data->Qy);
    readVar(is, "Qz", (void*)data->Qz);

    readVar(is, "Jx", (void*)data->Jx);
    readVar(is, "Jy", (void*)data->Jy);
    readVar(is, "Jz", (void*)data->Jz);

    readVar(is, (NB_PARTICLES_LABEL + SORT_0_LABEL).c_str(), (void *)&data->ionTotal);
    readVar(is, (CHARGE_LABEL + SORT_0_LABEL).c_str(), (void *)&data->chargeIons);
    readVar(is, (MASS_LABEL + SORT_0_LABEL).c_str(), (void *)&data->massIons);

    data->ionsX = new double[data->ionTotal];
    data->ionsY = new double[data->ionTotal];
    data->ionsZ = new double[data->ionTotal];

    data->ionsPx = new double[data->ionTotal];
    data->ionsPy = new double[data->ionTotal];
    data->ionsPz = new double[data->ionTotal];

    readVar(is, (COORDINATES_LABEL + X_LABEL + SORT_0_LABEL).c_str(), (void *) data->ionsX);
    readVar(is, (COORDINATES_LABEL + Y_LABEL + SORT_0_LABEL).c_str(), (void *) data->ionsY);
    readVar(is, (COORDINATES_LABEL + Z_LABEL + SORT_0_LABEL).c_str(), (void *) data->ionsZ);

    readVar(is, (IMPULSES_LABEL + X_LABEL + SORT_0_LABEL).c_str(), (void *) data->ionsPx);
    readVar(is, (IMPULSES_LABEL + Y_LABEL + SORT_0_LABEL).c_str(), (void *) data->ionsPy);
    readVar(is, (IMPULSES_LABEL + Z_LABEL + SORT_0_LABEL).c_str(), (void *) data->ionsPz);

    readVar(is, (NB_PARTICLES_LABEL + SORT_1_LABEL).c_str(), (void *)&data->electronsTotal);
    readVar(is, (CHARGE_LABEL + SORT_1_LABEL).c_str(), (void *)&data->chargeElectrons);
    readVar(is, (MASS_LABEL + SORT_1_LABEL).c_str(), (void *)&data->massElectrons);

    data->electronsX = new double[data->electronsTotal];
    data->electronsY = new double[data->electronsTotal];
    data->electronsZ = new double[data->electronsTotal];

    data->electronsPx = new double[data->electronsTotal];
    data->electronsPy = new double[data->electronsTotal];
    data->electronsPz = new double[data->electronsTotal];

    readVar(is, (COORDINATES_LABEL + X_LABEL + SORT_1_LABEL).c_str(), (void *) data->electronsX);
    readVar(is, (COORDINATES_LABEL + Y_LABEL + SORT_1_LABEL).c_str(), (void *) data->electronsY);
    readVar(is, (COORDINATES_LABEL + Z_LABEL + SORT_1_LABEL).c_str(), (void *) data->electronsZ);

    readVar(is, (IMPULSES_LABEL + X_LABEL + SORT_1_LABEL).c_str(), (void *) data->electronsPx);
    readVar(is, (IMPULSES_LABEL + Y_LABEL + SORT_1_LABEL).c_str(), (void *) data->electronsPy);
    readVar(is, (IMPULSES_LABEL + Z_LABEL + SORT_1_LABEL).c_str(), (void *) data->electronsPz);

    readVar(is, (NB_PARTICLES_LABEL + SORT_2_LABEL).c_str(), (void *)&data->beamTotal);
    readVar(is, (CHARGE_LABEL + SORT_2_LABEL).c_str(), (void *)&data->chargeBeam);
    readVar(is, (MASS_LABEL + SORT_2_LABEL).c_str(), (void *)&data->massBeam);

    data->beamX = new double[data->beamTotal];
    data->beamY = new double[data->beamTotal];
    data->beamZ = new double[data->beamTotal];

    data->beamPx = new double[data->beamTotal];
    data->beamPy = new double[data->beamTotal];
    data->beamPz = new double[data->beamTotal];

    readVar(is, (COORDINATES_LABEL + X_LABEL + SORT_2_LABEL).c_str(), (void *) data->beamX);
    readVar(is, (COORDINATES_LABEL + Y_LABEL + SORT_2_LABEL).c_str(), (void *) data->beamY);
    readVar(is, (COORDINATES_LABEL + Z_LABEL + SORT_2_LABEL).c_str(), (void *) data->beamZ);

    readVar(is, (IMPULSES_LABEL + X_LABEL + SORT_2_LABEL).c_str(), (void *) data->beamPx);
    readVar(is, (IMPULSES_LABEL + Y_LABEL + SORT_2_LABEL).c_str(), (void *) data->beamPy);
    readVar(is, (IMPULSES_LABEL + Z_LABEL + SORT_2_LABEL).c_str(), (void *) data->beamPz);

    return data;
}



