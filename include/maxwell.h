/*
 * init.h
 *
 *  Created on: Apr 14, 2016
 *      Author: snytav
 */

#ifndef MAXWELL_H_
#define MAXWELL_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <iostream>

#include "particle.h"
#include "run_control.h"
#include "archAPI.h"

typedef struct ParticleArrays {
    double *dbg_x, *dbg_y, *dbg_z, *dbg_px, *dbg_py, *dbg_pz, q_m, *m;
    int total;
} ParticleArrays;

void AllocateBinaryParticlesArrays(ParticleArrays *ions, ParticleArrays *electrons, ParticleArrays *beam_electrons);

double rnd_gaussian(double, double, int);

int getMassCharge(ParticleArrays *ions, ParticleArrays *electrons, ParticleArrays *beam_electrons, double ni, double rbd, int lp);

int InitUniformMaxwellianParticles(int beamf, int jmb,
                                   double tex0, double tey0, double tez0,
                                   double beam_lx, double beam_ly, double beam_lz,
                                   int *jmb_real,
                                   double lx, double ly, double lz,
                                   int meh, double Tb, double rimp, double rbd,
                                   double *xi, double *yi, double *zi, double *ui, double *vi, double *wi,
                                   double *xb, double *yb, double *zb, double *ub, double *vb, double *wb,
                                   double *xf, double *yf, double *zf, double *uf, double *vf, double *wf
);

int getUniformMaxwellianParticles(std::vector <Particle> &ion_vp,
                                  std::vector <Particle> &el_vp,
                                  std::vector <Particle> &beam_vp,
                                  double tex0, double tey0, double tez0,
                                  double Tb, double rimp, double rbd,
                                  double ni, int lp, int meh,
                                  double lx, double ly, double lz,
                                  int nx, int ny, int nz);

// todo: список параметров заменить на ParticleArrays
int convertParticleArraysToSTLvector(
        double *dbg_x,
        double *dbg_y,
        double *dbg_z,
        double *dbg_px,
        double *dbg_py,
        double *dbg_pz,
        double q_m,
        double m,
        int total_particles,
        particle_sorts sort,
        std::vector <Particle> &vp
);

int AllocateBinaryParticleArraysOneSort(double **dbg_x, double **dbg_y, double **dbg_z, double **dbg_px, double **dbg_py, double **dbg_pz, double **m, int total_particles);

#endif /* INIT_H_ */
