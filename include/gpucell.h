/*
 * gpucell.h
 *
 *  Created on: Oct 19, 2013
 *      Author: snytav
 */

#ifndef GPUCELL_H_
#define GPUCELL_H_
#ifdef __CUDACC__
#define host_device __host__ __device__
#else
#define host_device
#endif

#include "cell.h"
#include "archAPI.h"

void dbgPrintGPUParticleAttribute(Cell *d_c, int n_particle, int attribute, char *name);

__global__ void testKernel(double *vec);

class GPUCell : public Cell {
public:

host_device
    GPUCell();

host_device
    ~GPUCell();

host_device
    GPUCell(int i1, int l1, int k1, double Lx, double Ly, double Lz, int Nx1, int Ny1, int Nz1, double tau1);

    GPUCell *copyCellToDevice();

    void copyCellFromDevice(GPUCell *d_src, GPUCell *h_dst, std::string where, int nt);

    GPUCell *allocateCopyCellFromDevice();

    double compareToCell(Cell &d_src);
};

#endif /* GPUCELL_H_ */
