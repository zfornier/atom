/*
 * read_particles.h
 *
 *  Created on: Apr 9, 2016
 *      Author: snytav
 */

#ifndef READ_PARTICLES_H_
#define READ_PARTICLES_H_

#include <stdio.h>

#define SORTS 3

typedef struct ParticleArrays {
    double *dbg_x, *dbg_y, *dbg_z, *dbg_px, *dbg_py, *dbg_pz, q_m, *m;
    int total;
} ParticleArrays;

typedef struct ParticleFloatArrays {
    float *dbg_x, *dbg_y, *dbg_z, *dbg_px, *dbg_py, *dbg_pz, q_m, m;
    int total;
} ParticleFloatArrays;

typedef ParticleArrays ParticleArraysGroup[SORTS];

typedef ParticleFloatArrays ParticleFloatArraysGroup[SORTS];

void AllocateBinaryParticlesArrays(ParticleArrays *ions, ParticleArrays *electrons, ParticleArrays *beam_electrons);

#endif /* READ_PARTICLES_H_ */
