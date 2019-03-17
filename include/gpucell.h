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

void dbgPrintGPUParticleAttribute(Cell *d_c, int n_particle, int attribute, char *name);

__global__ void testKernel(double *vec);

class GPUCell : public Cell {
public:

#ifdef __CUDACC__
    __host__ __device__
#endif
    GPUCell() {}

#ifdef __CUDACC__
    __host__ __device__
#endif
    ~GPUCell() {}

#ifdef __CUDACC__
    __host__ __device__
#endif
    GPUCell(int i1, int l1, int k1, double Lx, double Ly, double Lz, int Nx1, int Ny1, int Nz1, double tau1) : Cell(i1, l1, k1, Lx, Ly, Lz, Nx1, Ny1, Nz1, tau1) {}

    GPUCell *copyCellToDevice() {
        GPUCell *h_src, *d_dst;
        int err1, err2, err3, err4, err5, err6, err7, err8, err9, err10;
        int err11, err12, err13, err14, err15, err16, err17, err18, err19, err20;
        int err21, err22, err23, err24, err25;

        h_src = new GPUCell;

        h_src->number_of_particles = Cell::number_of_particles;
        h_src->Nx = Cell::Nx;
        h_src->Ny = Cell::Ny;
        h_src->Nz = Cell::Nz;
        h_src->hx = Cell::hx;
        h_src->hy = Cell::hy;
        h_src->hz = Cell::hz;
        h_src->i = Cell::i;
        h_src->k = Cell::k;
        h_src->l = Cell::l;
        h_src->x0 = Cell::x0;
        h_src->y0 = Cell::y0;
        h_src->z0 = Cell::z0;
        h_src->xm = Cell::xm;
        h_src->ym = Cell::ym;
        h_src->zm = Cell::zm;
        h_src->tau = Cell::tau;
        h_src->jmp = Cell::jmp;
        h_src->d_ctrlParticles = Cell::d_ctrlParticles;

        h_src->busyParticleArray = Cell::busyParticleArray;

        cudaMalloc((void **) &(h_src->doubParticleArray), sizeof(Particle) * MAX_particles_per_cell);
        err1 = getLastError();


        cudaMemset((void **) h_src->doubParticleArray, 0, sizeof(Particle) * MAX_particles_per_cell);
        err2 = getLastError();

        MemoryCopy(h_src->doubParticleArray, Cell::doubParticleArray,
                   sizeof(Particle) * MAX_particles_per_cell, HOST_TO_DEVICE);
        err3 = getLastError();

        MemoryAllocate((void **) &(h_src->Jx), sizeof(CellDouble));
        err4 = getLastError();

        MemoryCopy(h_src->Jx, Cell::Jx, sizeof(CellDouble), HOST_TO_DEVICE);
        err5 = getLastError();

        MemoryAllocate((void **) &(h_src->Jy), sizeof(CellDouble));
        err6 = getLastError();

        MemoryCopy(h_src->Jy, Cell::Jy, sizeof(CellDouble), HOST_TO_DEVICE);
        err7 = getLastError();


        MemoryAllocate((void **) &(h_src->Jz), sizeof(CellDouble));
        err8 = getLastError();

        MemoryCopy(h_src->Jz, Cell::Jz, sizeof(CellDouble), HOST_TO_DEVICE);
        err9 = getLastError();

        MemoryAllocate((void **) &(h_src->Ex), sizeof(CellDouble));
        err10 = getLastError();

        MemoryCopy(h_src->Ex, Cell::Ex, sizeof(CellDouble), HOST_TO_DEVICE);
        err11 = getLastError();

        MemoryAllocate((void **) &(h_src->Ey), sizeof(CellDouble));
        err12 = getLastError();

        MemoryCopy(h_src->Ey, Cell::Ey, sizeof(CellDouble), HOST_TO_DEVICE);
        err13 = getLastError();

        MemoryAllocate((void **) &(h_src->Ez), sizeof(CellDouble));
        err14 = getLastError();

        MemoryCopy(h_src->Ez, Cell::Ez, sizeof(CellDouble), HOST_TO_DEVICE);
        err15 = getLastError();

        MemoryAllocate((void **) &(h_src->Hx), sizeof(CellDouble));
        err16 = getLastError();

        MemoryCopy(h_src->Hx, Cell::Hx, sizeof(CellDouble), HOST_TO_DEVICE);
        err17 = getLastError();

        cudaMalloc((void **) &(h_src->Hy), sizeof(CellDouble));
        err18 = getLastError();

        MemoryCopy(h_src->Hy, Cell::Hy, sizeof(CellDouble), HOST_TO_DEVICE);
        err19 = getLastError();

        MemoryAllocate((void **) &(h_src->Hz), sizeof(CellDouble));
        err20 = getLastError();

        MemoryCopy(h_src->Hz, Cell::Hz, sizeof(CellDouble), HOST_TO_DEVICE);
        err21 = getLastError();

        MemoryAllocate((void **) &(h_src->Rho), sizeof(CellDouble));
        err22 = getLastError();

        MemoryCopy(h_src->Rho, Cell::Rho, sizeof(CellDouble), HOST_TO_DEVICE);
        err23 = getLastError();

        MemoryAllocate((void **) &d_dst, sizeof(GPUCell));
        err24 = getLastError();

        MemoryCopy(d_dst, h_src, sizeof(GPUCell), HOST_TO_DEVICE);
        err25 = getLastError();

        if (
                (err1 != 0) ||
                (err2 != 0) ||
                (err3 != 0) ||
                (err4 != 0) ||
                (err5 != 0) ||
                (err6 != 0) ||
                (err7 != 0) ||
                (err8 != 0) ||
                (err9 != 0) ||
                (err10 != 0) ||
                (err11 != 0) ||
                (err12 != 0) ||
                (err13 != 0) ||
                (err14 != 0) ||
                (err15 != 0) ||
                (err16 != 0) ||
                (err17 != 0) ||
                (err18 != 0) ||
                (err19 != 0) ||
                (err20 != 0) ||
                (err21 != 0) ||
                (err22 != 0) ||
                (err23 != 0) ||
                (err24 != 0) ||
                (err25 != 0)
                ) {
            printf("copyCellToDevice error d_dst %p\n", d_dst);
        }

        return d_dst;
    }

    void copyCellFromDevice(GPUCell *d_src, GPUCell *h_dst, std::string where, int nt) {
        static GPUCell *h_copy_of_d_src;
        static int first = 1;
        int code;

        if (first == 1) {
            first = 0;
            h_copy_of_d_src = new GPUCell;
            h_copy_of_d_src->Init();

        }

        int err = getLastError();
        if (err != 0) {
            printf(" copyCellFromDevice enter %d %s \n ", err, getErrorString(err));
            exit(0);
        }

        cudaThreadSynchronize();

        err = MemoryCopy(h_copy_of_d_src, d_src, sizeof(GPUCell), DEVICE_TO_HOST);
        if (err != 0) {
            printf(" copyCellFromDevice1 %d %s \n ", err, getErrorString(err));
            exit(0);
        }
        if (h_copy_of_d_src->number_of_particles < 0 || h_copy_of_d_src->number_of_particles > MAX_particles_per_cell) {
        }
#ifdef COPY_CELLS_MEMORY_PRINTS
        printf("step %d %s number of particles %5d %3d %3d %d \n",nt,where.c_str(),h_copy_of_d_src->i,h_copy_of_d_src->l,h_copy_of_d_src->k,h_copy_of_d_src->number_of_particles);
#endif

        h_dst->number_of_particles = h_copy_of_d_src->number_of_particles;

        code = MemoryCopy(h_dst->doubParticleArray, h_copy_of_d_src->doubParticleArray, sizeof(Particle) * MAX_particles_per_cell, DEVICE_TO_HOST);
        if (code != 0) {
            printf(" copyCellFromDevice3 %d %s \n ", code, getErrorString(code));
            exit(0);
        }

        code = MemoryCopy(h_dst->Jx, h_copy_of_d_src->Jx, sizeof(CellDouble), DEVICE_TO_HOST);
        if (code != 0) {
            printf(" copyCellFromDevice4 %d \n ", code);
            exit(0);
        }

        code = MemoryCopy(h_dst->Jy, h_copy_of_d_src->Jy, sizeof(CellDouble), DEVICE_TO_HOST);
        if (code != 0) {
            printf(" copyCellFromDevice5 %d \n ", code);
            exit(0);
        }

        code = MemoryCopy(h_dst->Jz, h_copy_of_d_src->Jz, sizeof(CellDouble), DEVICE_TO_HOST);
        if (code != 0) {
            printf(" copyCellFromDevice6 %d \n ", code);
            exit(0);
        }

        code = MemoryCopy(h_dst->Ex, h_copy_of_d_src->Ex, sizeof(CellDouble), DEVICE_TO_HOST);
        code = MemoryCopy(h_dst->Ey, h_copy_of_d_src->Ey, sizeof(CellDouble), DEVICE_TO_HOST);
        code = MemoryCopy(h_dst->Ez, h_copy_of_d_src->Ez, sizeof(CellDouble), DEVICE_TO_HOST);
        code = MemoryCopy(h_dst->Hx, h_copy_of_d_src->Hx, sizeof(CellDouble), DEVICE_TO_HOST);
        code = MemoryCopy(h_dst->Hy, h_copy_of_d_src->Hy, sizeof(CellDouble), DEVICE_TO_HOST);
        code = MemoryCopy(h_dst->Hz, h_copy_of_d_src->Hz, sizeof(CellDouble), DEVICE_TO_HOST);
        code = MemoryCopy(h_dst->Rho, h_copy_of_d_src->Rho, sizeof(CellDouble), DEVICE_TO_HOST);

        if (code != 0) {
            printf(" copyCellFromDevice10 %d \n ", code);
            exit(0);
        }
    }

    GPUCell *allocateCopyCellFromDevice() {
        GPUCell *h_dst;

        h_dst = new GPUCell;

        h_dst->doubParticleArray = (double *) malloc(sizeof(Particle) * MAX_particles_per_cell);

        h_dst->Jx = new CellDouble;
        h_dst->Jy = new CellDouble;
        h_dst->Jz = new CellDouble;

        h_dst->Ex = new CellDouble;
        h_dst->Ey = new CellDouble;
        h_dst->Ez = new CellDouble;
        h_dst->Hx = new CellDouble;
        h_dst->Hy = new CellDouble;
        h_dst->Hz = new CellDouble;
        h_dst->Rho = new CellDouble;

        return h_dst;
    }

    double compareToCell(Cell &d_src) {
        return Cell::compareToCell(d_src);
    }

};

#endif /* GPUCELL_H_ */
