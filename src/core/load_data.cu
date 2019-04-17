/*
 * read_particle.cxx
 *
 *  Created on: Jun 9, 2018
 *      Author: snytav
 */

#include "../../include/load_data.h"

void debugPrintParticleCharacteristicArray(double *p_ch, int np, int nt, char *name, int sort) {

#ifndef PRINT_PARTICLE_INITIALS
    return;
#else
    sprintf(fname,"particle_init_%s_%05d_sort%02d.dat",name,nt,sort);

    if((f = fopen(fname,"wt")) == NULL) return;

    for (int i = 0;i < np;i++) {
        fprintf(f,"%10d %10d %25.16e \n",i,i+1,p_ch[i]);
    }

    fclose(f);
#endif
}

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

void readBinaryParticleArraysOneSort(
        const char * filename,
        double *dbg_x, double *dbg_y, double *dbg_z, double *dbg_px, double *dbg_py, double *dbg_pz,
        int total_particles,
        int nt,
        int sort
) {

    //Reading X coordinates for particles of sort "sort"
    readVar(filename, (std::string("Coordinates_x") + patch::to_string(sort)).c_str(), (void *) dbg_x);

    //Reading Y coordinates for particles of sort "sort"
    readVar(filename, (std::string("Coordinates_y") + patch::to_string(sort)).c_str(), (void *) dbg_y);

    //Reading Z coordinates for particles of sort "sort"
    readVar(filename, (std::string("Coordinates_z") + patch::to_string(sort)).c_str(), (void *) dbg_z);

    //Reading X impulses for particles of sort "sort"
    readVar(filename, (std::string("Impulses_x") + patch::to_string(sort)).c_str(), (void *) dbg_px);

    //Reading Y impulses for particles of sort "sort"
    readVar(filename, (std::string("Impulses_y") + patch::to_string(sort)).c_str(), (void *) dbg_py);

    //Reading Z impulses for particles of sort "sort"
    readVar(filename, (std::string("Impulses_z") + patch::to_string(sort)).c_str(), (void *) dbg_pz);

    debugPrintParticleCharacteristicArray(dbg_x, total_particles, nt, (char*)"x", sort);
    debugPrintParticleCharacteristicArray(dbg_y, total_particles, nt, (char*)"y", sort);
    debugPrintParticleCharacteristicArray(dbg_z, total_particles, nt, (char*)"z", sort);
    debugPrintParticleCharacteristicArray(dbg_px, total_particles, nt, (char*)"px", sort);
    debugPrintParticleCharacteristicArray(dbg_py, total_particles, nt, (char*)"py", sort);
    debugPrintParticleCharacteristicArray(dbg_pz, total_particles, nt, (char*)"pz", sort);

}



