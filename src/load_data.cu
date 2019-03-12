/*
 * read_particle.cxx
 *
 *  Created on: Jun 9, 2018
 *      Author: snytav
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <stdint.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <iostream>
#include <string>

#include "../include/load_data.h"

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

int readBinaryParticleArraysOneSort(
        FILE *f,
        double **dbg_x,
        double **dbg_y,
        double **dbg_z,
        double **dbg_px,
        double **dbg_py,
        double **dbg_pz,
        double *qq_m,
        double *mm,
        int nt,
        int sort
) {
    double q_m, m;
    int t;
    int total_particles;
    int err;

    if ((err = ferror(f)) != 0) {
        return err;
    }

    /*
* name + sort
* names:
    * Extra_number_    int
    * Nb_particles_    int
    * Charge_  	     double
    * Mass_            double
    * Coordinates_x    double
    * Coordinates_y	 double
    * Coordinates_z	 double
    * Impulses_x		 double
    * Impulses_y		 double
    * Impulses_z		 double
* func: 	readVar("filename.nc", (std::to_string("Name") +  std::to_string(sort)).c_str(), &var);
* filename: mumu60000000005.nc
*/
    //Reading extra number placed by Fortran

    readVar("mumu60000000005.nc", (std::string("Extra_number_") + patch::to_string(sort)).c_str(), &t);

    //Reading number of particles of sort "sort"
    readVar("mumu60000000005.nc", (std::string("Nb_particles_") + patch::to_string(sort)).c_str(), &total_particles);

    //Reading charge for sort "sort"

    readVar("mumu60000000005.nc", (std::string("Charge_") + patch::to_string(sort)).c_str(), &q_m);

    //Reading mass for sort "sort"
    readVar("mumu60000000005.nc", (std::string("Mass_") + patch::to_string(sort)).c_str(), &m);

    // Reading extra number placed by Fortran
    readVar("mumu60000000005.nc", (std::string("Extra_number_") + patch::to_string(sort)).c_str(), &t);

    double *dbg_x1 = (double *) malloc(sizeof(double) * total_particles);
    double *dbg_y1 = (double *) malloc(sizeof(double) * total_particles);
    double *dbg_z1 = (double *) malloc(sizeof(double) * total_particles);
    double *dbg_px1 = (double *) malloc(sizeof(double) * total_particles);
    double *dbg_py1 = (double *) malloc(sizeof(double) * total_particles);
    double *dbg_pz1 = (double *) malloc(sizeof(double) * total_particles);

    //Reading X coordinates for particles of sort "sort"
    readVar("mumu60000000005.nc", (std::string("Coordinates_x") + patch::to_string(sort)).c_str(), (void *) dbg_x1);

    //Reading Y coordinates for particles of sort "sort"
    readVar("mumu60000000005.nc", (std::string("Coordinates_y") + patch::to_string(sort)).c_str(), (void *) dbg_y1);

    //Reading Z coordinates for particles of sort "sort"
    readVar("mumu60000000005.nc", (std::string("Coordinates_z") + patch::to_string(sort)).c_str(), (void *) dbg_z1);

    //Reading X impulses for particles of sort "sort"
    readVar("mumu60000000005.nc", (std::string("Impulses_x") + patch::to_string(sort)).c_str(), (void *) dbg_px1);

    //Reading Y impulses for particles of sort "sort"
    readVar("mumu60000000005.nc", (std::string("Impulses_y") + patch::to_string(sort)).c_str(), (void *) dbg_py1);

    //Reading Z impulses for particles of sort "sort"
    readVar("mumu60000000005.nc", (std::string("Impulses_z") + patch::to_string(sort)).c_str(), (void *) dbg_pz1);

    *dbg_x = dbg_x1;
    *dbg_y = dbg_y1;
    *dbg_z = dbg_z1;

    *dbg_px = dbg_px1;
    *dbg_py = dbg_py1;
    *dbg_pz = dbg_pz1;

    debugPrintParticleCharacteristicArray(*dbg_x, total_particles, nt, "x", sort);
    debugPrintParticleCharacteristicArray(*dbg_y, total_particles, nt, "y", sort);
    debugPrintParticleCharacteristicArray(*dbg_z, total_particles, nt, "z", sort);
    debugPrintParticleCharacteristicArray(*dbg_px, total_particles, nt, "px", sort);
    debugPrintParticleCharacteristicArray(*dbg_py, total_particles, nt, "py", sort);
    debugPrintParticleCharacteristicArray(*dbg_pz, total_particles, nt, "pz", sort);

    *qq_m = q_m;
    *mm = m;

    if ((err = ferror(f)) != 0) {
        return err;
    }

    return total_particles;
}



