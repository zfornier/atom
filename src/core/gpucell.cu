/*
 * gpucell.cu
 *
 *  Created on: Jul 23, 2018
 *      Author: snytav
 */

#include "../../include/gpucell.h"

__host__ __device__
GPUCell::GPUCell() {}

__host__ __device__
GPUCell::~GPUCell() {}

__host__ __device__
GPUCell::GPUCell(int i1, int l1, int k1, double Lx, double Ly, double Lz, int Nx1, int Ny1, int Nz1, double tau1) : Cell(i1, l1, k1, Lx, Ly, Lz, Nx1, Ny1, Nz1, tau1) {}

GPUCell *GPUCell::copyCellToDevice() {
    GPUCell *h_src, *d_dst;
    int err;

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

    err = cudaMalloc((void **) &(h_src->doubParticleArray), sizeof(Particle) * MAX_particles_per_cell);
    CHECK_ERROR("", err);

    err = cudaMemset((void **) h_src->doubParticleArray, 0, sizeof(Particle) * MAX_particles_per_cell);
    CHECK_ERROR("", err);

    err = MemoryCopy(h_src->doubParticleArray, Cell::doubParticleArray, sizeof(Particle) * MAX_particles_per_cell, HOST_TO_DEVICE);
    CHECK_ERROR("", err);

    err = MemoryAllocate((void **) &(h_src->Jx), sizeof(CellDouble));
    CHECK_ERROR("", err);

    err = MemoryCopy(h_src->Jx, Cell::Jx, sizeof(CellDouble), HOST_TO_DEVICE);
    CHECK_ERROR("", err);

    err = MemoryAllocate((void **) &(h_src->Jy), sizeof(CellDouble));
    CHECK_ERROR("", err);

    err = MemoryCopy(h_src->Jy, Cell::Jy, sizeof(CellDouble), HOST_TO_DEVICE);
    CHECK_ERROR("", err);

    err = MemoryAllocate((void **) &(h_src->Jz), sizeof(CellDouble));
    CHECK_ERROR("", err);

    err = MemoryCopy(h_src->Jz, Cell::Jz, sizeof(CellDouble), HOST_TO_DEVICE);
    CHECK_ERROR("", err);

    err = MemoryAllocate((void **) &(h_src->Ex), sizeof(CellDouble));
    CHECK_ERROR("", err);

    err = MemoryCopy(h_src->Ex, Cell::Ex, sizeof(CellDouble), HOST_TO_DEVICE);
    CHECK_ERROR("", err);

    err = MemoryAllocate((void **) &(h_src->Ey), sizeof(CellDouble));
    CHECK_ERROR("", err);

    err = MemoryCopy(h_src->Ey, Cell::Ey, sizeof(CellDouble), HOST_TO_DEVICE);
    CHECK_ERROR("", err);

    err = MemoryAllocate((void **) &(h_src->Ez), sizeof(CellDouble));
    CHECK_ERROR("", err);

    err = MemoryCopy(h_src->Ez, Cell::Ez, sizeof(CellDouble), HOST_TO_DEVICE);
    CHECK_ERROR("", err);

    err = MemoryAllocate((void **) &(h_src->Hx), sizeof(CellDouble));
    CHECK_ERROR("", err);

    err = MemoryCopy(h_src->Hx, Cell::Hx, sizeof(CellDouble), HOST_TO_DEVICE);
    CHECK_ERROR("", err);

    err = cudaMalloc((void **) &(h_src->Hy), sizeof(CellDouble));
    CHECK_ERROR("", err);

    err = MemoryCopy(h_src->Hy, Cell::Hy, sizeof(CellDouble), HOST_TO_DEVICE);
    CHECK_ERROR("", err);

    err = MemoryAllocate((void **) &(h_src->Hz), sizeof(CellDouble));
    CHECK_ERROR("", err);

    err = MemoryCopy(h_src->Hz, Cell::Hz, sizeof(CellDouble), HOST_TO_DEVICE);
    CHECK_ERROR("", err);

    err = MemoryAllocate((void **) &(h_src->Rho), sizeof(CellDouble));
    CHECK_ERROR("", err);

    err = MemoryCopy(h_src->Rho, Cell::Rho, sizeof(CellDouble), HOST_TO_DEVICE);
    CHECK_ERROR("", err);

    err = MemoryAllocate((void **) &d_dst, sizeof(GPUCell));
    CHECK_ERROR("", err);

    err = MemoryCopy(d_dst, h_src, sizeof(GPUCell), HOST_TO_DEVICE);
    CHECK_ERROR("", err);

    return d_dst;
}

void GPUCell::copyCellFromDevice(GPUCell *d_src, GPUCell *h_dst) {
    static GPUCell *h_copy_of_d_src;
    static int first = 1;
    int err;

    if (first == 1) {
        first = 0;
        h_copy_of_d_src = new GPUCell;
        h_copy_of_d_src->Init();
    }

    err = cudaThreadSynchronize();
    CHECK_ERROR("Thread synchronize", err);

    err = MemoryCopy(h_copy_of_d_src, d_src, sizeof(GPUCell), DEVICE_TO_HOST);
    CHECK_ERROR("copyCellFromDevice1", err);

    h_dst->number_of_particles = h_copy_of_d_src->number_of_particles;

    err = MemoryCopy(h_dst->doubParticleArray, h_copy_of_d_src->doubParticleArray, sizeof(Particle) * MAX_particles_per_cell, DEVICE_TO_HOST);
    CHECK_ERROR("copyCellFromDevice2", err);

    err = MemoryCopy(h_dst->Jx, h_copy_of_d_src->Jx, sizeof(CellDouble), DEVICE_TO_HOST);
    CHECK_ERROR("copyCellFromDevice3", err);

    err = MemoryCopy(h_dst->Jy, h_copy_of_d_src->Jy, sizeof(CellDouble), DEVICE_TO_HOST);
    CHECK_ERROR("copyCellFromDevice4", err);

    err = MemoryCopy(h_dst->Jz, h_copy_of_d_src->Jz, sizeof(CellDouble), DEVICE_TO_HOST);
    CHECK_ERROR("copyCellFromDevice5", err);

    err = MemoryCopy(h_dst->Ex, h_copy_of_d_src->Ex, sizeof(CellDouble), DEVICE_TO_HOST);
    CHECK_ERROR("copyCellFromDevice6", err);

    err = MemoryCopy(h_dst->Ey, h_copy_of_d_src->Ey, sizeof(CellDouble), DEVICE_TO_HOST);
    CHECK_ERROR("copyCellFromDevice7", err);

    err = MemoryCopy(h_dst->Ez, h_copy_of_d_src->Ez, sizeof(CellDouble), DEVICE_TO_HOST);
    CHECK_ERROR("copyCellFromDevice8", err);

    err = MemoryCopy(h_dst->Hx, h_copy_of_d_src->Hx, sizeof(CellDouble), DEVICE_TO_HOST);
    CHECK_ERROR("copyCellFromDevice9", err);

    err = MemoryCopy(h_dst->Hy, h_copy_of_d_src->Hy, sizeof(CellDouble), DEVICE_TO_HOST);
    CHECK_ERROR("copyCellFromDevice10", err);

    err = MemoryCopy(h_dst->Hz, h_copy_of_d_src->Hz, sizeof(CellDouble), DEVICE_TO_HOST);
    CHECK_ERROR("copyCellFromDevice11", err);

    err = MemoryCopy(h_dst->Rho, h_copy_of_d_src->Rho, sizeof(CellDouble), DEVICE_TO_HOST);
    CHECK_ERROR("copyCellFromDevice12", err);
}

GPUCell *GPUCell::allocateCopyCellFromDevice() {
    GPUCell *h_dst;

    h_dst = new GPUCell;
    int size = sizeof(Particle) / sizeof(double);
    h_dst->doubParticleArray = new double[size * MAX_particles_per_cell];

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

double GPUCell::compareToCell(Cell &d_src) {
    return Cell::compareToCell(d_src);
}

