/*
 * init.h
 *
 *  Created on: Apr 14, 2016
 *      Author: snytav
 */

#ifndef MAXWELL_H_
#define MAXWELL_H_

#include <vector>
#include <iostream>

#include "particle.h"
#include "run_control.h"
#include "archAPI.h"

typedef struct ParticleArrays {
    double *x, *y, *z, *px, *py, *pz, q_m, *m;
    int total;
} ParticleArrays;

typedef struct ParticlesConfig {
    ParticleArrays *ions, *electrons, *beam;
    double tempX, tempY, tempZ;
    double beamVelDisp, beamImp, beamPlasmaDensityRat, plsmDensity;
    int lp, meh;
    double lx, ly, lz;
    double beam_lx, beam_ly, beam_lz;
    int nx, ny, nz;
    int beamPlasma;
} ParticlesConfig;

void AllocateBinaryParticlesArrays(ParticleArrays *, ParticleArrays *, ParticleArrays *);

double rnd_gaussian(double, double, int);

int getMassCharge(ParticleArrays *, ParticleArrays *, ParticleArrays *, double, double, int);

int InitUniformMaxwellianParticles(ParticlesConfig *, int, int *);

int getUniformMaxwellianParticles(std::vector <Particle> &, std::vector <Particle> &, std::vector <Particle> &, ParticlesConfig *);

int convertParticleArraysToSTLvector(ParticleArrays *, particle_sorts sort, std::vector <Particle> &vp);

int AllocateBinaryParticleArraysOneSort(double **, double **, double **, double **, double **, double **, double **, int);

#endif /* INIT_H_ */
