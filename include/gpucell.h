/*
 * gpucell.h
 *
 *  Created on: Oct 19, 2013
 *      Author: snytav
 */

#ifndef GPUCELL_H_
#define GPUCELL_H_

#include "cell.h"
#include "archAPI.h"

class GPUCell : public Cell {
public:

    __host__ __device__
    GPUCell();

    __host__ __device__
    ~GPUCell();

    __host__ __device__
    GPUCell(int i1, int l1, int k1, double Lx, double Ly, double Lz, int Nx1, int Ny1, int Nz1, double tau1);

    GPUCell *copyCellToDevice();

    void copyCellFromDevice(GPUCell *d_src, GPUCell *h_dst);

    GPUCell *allocateCopyCellFromDevice();

    double compareToCell(Cell &d_src);
};

#endif /* GPUCELL_H_ */
