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



