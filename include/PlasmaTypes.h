//
// Created by egor on 19.03.19.
//

#ifndef ATOM_PLASMACONFIG_H
#define ATOM_PLASMACONFIG_H

#include <string>
#include "gpucell.h"

typedef struct {
    double tempX;                // plasma electron temperature along X
    double tempY;                // plasma electron temperature along Y
    double tempZ;                // plasma electron temperature along Z
    double beamImp;              // beam impulse - rimp
    double beamVelDisp;          // beam velocity dispersion - Tb
    double beamPlasmaDensityRat; // beam and plasma density ratio - rbd
    double plsmDensity;          // plasma density - ni
    double externalMagnFieldX;   // external magnetic field (along X)
    double lx;                   // domain size X
    double ly;                   // domain size Y
    double lz;                   // domain size Z
    double px;                   // plasma size X
    double py;                   // plasma size Y
    double bx;                   // beam size X
    double by;                   // beam size Y
    double bz;                   // beam size Z
    int lp;                      // average number of particles in cell
    int nx;                      // number of mesh nodes along X
    int ny;                      // number of mesh nodes along Y
    int nz;                      // number of mesh nodes along Z
    double tau;                  // timestep
    int beamPlasma;              // 1 if beam-plasma interaction, 0 if beam-beam
    int meh;                     // horizontal process number (with mixed decomposition)

    // Computation parameters
    int st;                      // start step
    int lst;                     // last step
    int saveStep;                // save every saveStep step
    int startSave;               // start save from startSave step
    const char * checkFile;      // file to check with

    int jmp;
    int total_particles;
    double ami, amb, amf;
    GPUCell **h_CellArray, **d_CellArray;
    GPUCell **cp;
    double *d_Ex, *d_Ey, *d_Ez, *d_Hx, *d_Hy, *d_Hz, *d_Jx, *d_Jy, *d_Jz, *d_Rho, *d_npJx, *d_npJy, *d_npJz;
    double *d_Qx, *d_Qy, *d_Qz;
    double *dbg_x, *dbg_y, *dbg_z, *dbg_px, *dbg_py, *dbg_pz;
    double *ctrlParticles, *d_ctrlParticles;
    double *Qx, *Qy, *Qz, *dbg_Qx, *dbg_Qy, *dbg_Qz;
    double *Ex, *Ey, *Ez, *Hx, *Hy, *Hz, *Jx, *Jy, *Jz, *Rho, *npJx, *npJy, *npJz;
    double *dbgEx, *dbgEy, *dbgEz, *dbgHx, *dbgHy, *dbgHz, *dbgJx, *dbgJy, *dbgJz;
    std::vector <GPUCell> *AllCells;
    FILE *f_prec_report;
} PlasmaConfig;


#endif //ATOM_PLASMACONFIG_H
