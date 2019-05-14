/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include "../../include/mpi_shortcut.h"

int InitMPI(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    return 0;
}

int sumMPI(int size, double *d_jx, double *d_jy, double *d_jz) {
    double *snd, *rcv, *jx, *jy, *jz;
    int i;
    int err;

    jx = new double[size];
    jy = new double[size];
    jz = new double[size];

    err = MemoryCopy(jx, d_jx, size * sizeof(double), DEVICE_TO_HOST);
    CHECK_ERROR("Sum mpi", err);

    err = MemoryCopy(jy, d_jy, size * sizeof(double), DEVICE_TO_HOST);
    CHECK_ERROR("Sum mpi", err);

    err = MemoryCopy(jz, d_jz, size * sizeof(double), DEVICE_TO_HOST);
    CHECK_ERROR("Sum mpi", err);

    snd = new double[3 * size];
    rcv = new double[3 * size];

    for (i = 0; i < size; i++) {
        snd[i] = jx[i];
        snd[i + size] = jy[i];
        snd[i + 2 * size] = jz[i];
    }

    MPI_Allreduce(snd, rcv, size, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD);

    for (i = 0; i < size; i++) {
        jx[i] = rcv[i];
        jy[i] = rcv[i + size];
        jz[i] = rcv[i + 2 * size];
    }

    err = MemoryCopy(d_jx, jx, size * sizeof(double), HOST_TO_DEVICE);
    CHECK_ERROR("Sum mpi", err);

    err = MemoryCopy(d_jy, jy, size * sizeof(double), HOST_TO_DEVICE);
    CHECK_ERROR("Sum mpi", err);

    err = MemoryCopy(d_jz, jz, size * sizeof(double), HOST_TO_DEVICE);
    CHECK_ERROR("Sum mpi", err);

    delete[] jx;
    delete[] jy;
    delete[] jz;

    delete[] snd;
    delete[] rcv;

    return 0;
}

int sumMPIenergy(double *e) {
    double snd, rcv;

    snd = *e;

    MPI_Allreduce(&snd, &rcv, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD);

    *e = rcv;

    return 0;
}

int getRank() {
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    return rank;
}

int getSize() {
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &rank);

    return rank;
}

int CloseMPI() {
    MPI_Finalize();

    return 0;
}
