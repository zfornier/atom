/*
 * read_particles.h
 *
 *  Created on: Apr 9, 2016
 *      Author: snytav
 */

#ifndef READ_PARTICLES_H_
#define READ_PARTICLES_H_

typedef struct ParticleArrays {
    double *dbg_x, *dbg_y, *dbg_z, *dbg_px, *dbg_py, *dbg_pz, q_m, *m;
    int total;
} ParticleArrays;

void AllocateBinaryParticlesArrays(ParticleArrays *ions, ParticleArrays *electrons, ParticleArrays *beam_electrons);

#endif /* READ_PARTICLES_H_ */
