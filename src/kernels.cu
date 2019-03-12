
__global__ void GPU_SetAllCurrentsToZero(GPUCell **cells) {
    unsigned int nx = blockIdx.x;
    unsigned int ny = blockIdx.y;
    unsigned int nz = blockIdx.z;
    Cell *c, *c0 = cells[0], nc;

    c = cells[c0->getGlobalCellNumber(nx, ny, nz)];

    nc = *c;

    nc.SetAllCurrentsToZero(threadIdx);
}

__global__ void GPU_SetFieldsToCells(GPUCell **cells, double *Ex, double *Ey, double *Ez, double *Hx, double *Hy, double *Hz) {
    unsigned int nx = blockIdx.x;
    unsigned int ny = blockIdx.y;
    unsigned int nz = blockIdx.z;
    Cell *c, *c0 = cells[0];

    c = cells[c0->getGlobalCellNumber(nx, ny, nz)];


    c->readFieldsFromArrays(Ex, Ey, Ez, Hx, Hy, Hz, threadIdx);
}

__global__ void GPU_WriteAllCurrents(GPUCell **cells, int n0, double *jx, double *jy, double *jz, double *rho) {
    unsigned int nx = blockIdx.x;
    unsigned int ny = blockIdx.y;
    unsigned int nz = blockIdx.z;
    Cell *c, *c0 = cells[0];

    c = cells[c0->getGlobalCellNumber(nx, ny, nz)];

    int i1, l1, k1;
    i1 = threadIdx.x;
    l1 = threadIdx.y;
    k1 = threadIdx.z;
    int n = c->getFortranCellNumber(c->i + i1 - 1, c->l + l1 - 1, c->k + k1 - 1);

    if (n < 0) n = -n;
    double t, t_x, t_y;
    t_x = c->Jx->M[i1][l1][k1];
    int3 i3 = c->getCellTripletNumber(n);

    cuda_atomicAdd(&(jx[n]), t_x);
    t_y = c->Jy->M[i1][l1][k1];
    cuda_atomicAdd(&(jy[n]), t_y);
    t = c->Jz->M[i1][l1][k1];
    cuda_atomicAdd(&(jz[n]), t);
}

__global__ void GPU_WriteControlSystem(Cell **cells) {
    unsigned int nx = blockIdx.x;
    unsigned int ny = blockIdx.y;
    unsigned int nz = blockIdx.z;
    Cell *c, *c0 = cells[0], nc;

    c = cells[c0->getGlobalCellNumber(nx, ny, nz)];

    nc = *c;

    nc.SetControlSystemToParticles();

}

//TODO : 1. 3 separate kernels :
//            A. form 3x3x3 array with number how many to fly and departure list with start positions in 3x3x3 array
//            B. func to get 3x3x3 indexes from a pair of cell numbers, to and from function
//            C. 2nd kernel to write arrival 3x3x3 matrices
///           D. 3rd kernel to form arrival positions in the particle list
//            E. 4th to write arriving particles


__global__ void GPU_MakeDepartureLists(GPUCell **cells, int nt, int *d_stage) {
    unsigned int nx = blockIdx.x;
    unsigned int ny = blockIdx.y;
    unsigned int nz = blockIdx.z;
    int ix, iy, iz;

    Particle p;
    Cell *c, *c0 = cells[0], nc;
    c = cells[c0->getGlobalCellNumber(nx, ny, nz)];

    c->departureListLength = 0;
    for (ix = 0; ix < 3; ix++) {
        for (iy = 0; iy < 3; iy++) {
            for (iz = 0; iz < 3; iz++) {
                c->departure[ix][iy][iz] = 0;
            }
        }
    }
    c->departureListLength = 0;
    for (int num = 0; num < c->number_of_particles; num++) {
        p = c->readParticleFromSurfaceDevice(num);

        if (!c->isPointInCell(p.GetX())) { //check Paricle = operator !!!!!!!!!!!!!!!!!!!!!!!!!!!
            c->removeParticleFromSurfaceDevice(num, &p, &(c->number_of_particles));
            c->flyDirection(&p, &ix, &iy, &iz);
            if (p.fortran_number == 325041 && p.sort == 2) {
                d_stage[0] = ix;
                d_stage[1] = iy;
                d_stage[2] = iz;
            }

//TODO: mke FINAL print at STRAY function.
//					Make 3x3x3x20(50) particle fly array at each cell

            //departureList[departureListLength++] = p;

            if (c->departureListLength == PARTICLES_FLYING_ONE_DIRECTION) {
                d_stage[0] = TOO_MANY_PARTICLES;
                d_stage[1] = c->i;
                d_stage[2] = c->l;
                d_stage[3] = c->k;
                d_stage[1] = ix;
                d_stage[2] = iy;
                d_stage[3] = iz;
                return;
            }
            c->departureListLength++;
            int num1 = c->departure[ix][iy][iz];

            c->departureList[ix][iy][iz][num1] = p;
            if (p.fortran_number == 325041 && p.sort == 2) {
                d_stage[4] = num1;
                d_stage[5] = c->departureList[ix][iy][iz][num1].fortran_number;

            }

            c->departure[ix][iy][iz] += 1;
            num--;
        }
    }
}

__global__ void GPU_ArrangeFlights(GPUCell **cells, int nt, int *d_stage) {
    unsigned int nx = blockIdx.x;
    unsigned int ny = blockIdx.y;
    unsigned int nz = blockIdx.z;
    int ix, iy, iz, snd_ix, snd_iy, snd_iz, num, n;
    Particle p;

    Cell *c, *c0 = cells[0], nc, *snd_c;

    c = cells[n = c0->getGlobalCellNumber(nx, ny, nz)];

    for (ix = 0; ix < 3; ix++)
        for (iy = 0; iy < 3; iy++)
            for (iz = 0; iz < 3; iz++) {
                int index = ix * 9 + iy * 3 + iz;
                n = c0->getWrapCellNumber(nx + ix - 1, ny + iy - 1, nz + iz - 1);

                snd_c = cells[n];
                if (nx == 24 && ny == 2 && nz == 2) {

                    d_stage[index * 4] = snd_c->i;
                    d_stage[index * 4 + 1] = snd_c->l;
                    d_stage[index * 4 + 2] = snd_c->k;
                    d_stage[index * 4 + 3] = snd_c->departureListLength;
                }

                snd_ix = ix;
                snd_iy = iy;
                snd_iz = iz;
                c->inverseDirection(&snd_ix, &snd_iy, &snd_iz);

                num = snd_c->departure[snd_ix][snd_iy][snd_iz];

                for (int i = 0; i < num; i++) {
                    p = snd_c->departureList[snd_ix][snd_iy][snd_iz][i];
                    if (nx == 24 && ny == 2 && nz == 2) {}
                    c->Insert(p);
                }
            }
}

__device__ void writeCurrentComponent(CellDouble *J, CurrentTensorComponent *t1, CurrentTensorComponent *t2, int pqr2) {
    cuda_atomicAdd(&(J->M[t1->i11][t1->i12][t1->i13]), t1->t[0]);
    cuda_atomicAdd(&(J->M[t1->i21][t1->i22][t1->i23]), t1->t[1]);
    cuda_atomicAdd(&(J->M[t1->i31][t1->i32][t1->i33]), t1->t[2]);
    cuda_atomicAdd(&(J->M[t1->i41][t1->i42][t1->i43]), t1->t[3]);

    if (pqr2 == 2) {
        cuda_atomicAdd(&(J->M[t2->i11][t2->i12][t2->i13]), t2->t[0]);
        cuda_atomicAdd(&(J->M[t2->i21][t2->i22][t2->i23]), t2->t[1]);
        cuda_atomicAdd(&(J->M[t2->i31][t2->i32][t2->i33]), t2->t[2]);
        cuda_atomicAdd(&(J->M[t2->i41][t2->i42][t2->i43]), t2->t[3]);
    }

}

__device__ void copyCellDouble(CellDouble *dst, CellDouble *src, unsigned int n, uint3 block) {
    if (n < CellExtent * CellExtent * CellExtent) {
        double *d_dst, *d_src;//,t;

        d_dst = (double *) (dst->M);
        d_src = (double *) (src->M);

        d_dst[n] = d_src[n];
    }
}

__device__ void setCellDoubleToZero(CellDouble *dst, unsigned int n) {
    if (n < CellExtent * CellExtent * CellExtent) {
        double *d_dst;

        d_dst = (double *) (dst->M);
        d_dst[n] = 0.0;
    }
}

__device__ void assignSharedWithLocal(
        CellDouble **c_jx,
        CellDouble **c_jy,
        CellDouble **c_jz,
        CellDouble **c_ex,
        CellDouble **c_ey,
        CellDouble **c_ez,
        CellDouble **c_hx,
        CellDouble **c_hy,
        CellDouble **c_hz,
        CellDouble *fd) {
    *c_ex = &(fd[0]);
    *c_ey = &(fd[1]);
    *c_ez = &(fd[2]);

    *c_hx = &(fd[3]);
    *c_hy = &(fd[4]);
    *c_hz = &(fd[5]);

    *c_jx = &(fd[6]);
    *c_jy = &(fd[7]);
    *c_jz = &(fd[8]);
}
__device__ void copyFieldsToSharedMemory(
        CellDouble *c_jx,
        CellDouble *c_jy,
        CellDouble *c_jz,
        CellDouble *c_ex,
        CellDouble *c_ey,
        CellDouble *c_ez,
        CellDouble *c_hx,
        CellDouble *c_hy,
        CellDouble *c_hz,
        Cell *c,
        int index,
        dim3 blockId,
        int blockDimX
) {

    while (index < CellExtent * CellExtent * CellExtent) {

        copyCellDouble(c_ex, c->Ex, index, blockId);
        copyCellDouble(c_ey, c->Ey, index, blockId);
        copyCellDouble(c_ez, c->Ez, index, blockId);

        copyCellDouble(c_hx, c->Hx, index, blockId);
        copyCellDouble(c_hy, c->Hy, index, blockId);
        copyCellDouble(c_hz, c->Hz, index, blockId);

        copyCellDouble(c_jx, c->Jx, index, blockId);
        copyCellDouble(c_jy, c->Jy, index, blockId);
        copyCellDouble(c_jz, c->Jz, index, blockId);
        index += blockDimX;
    }

    __syncthreads();

}

__device__ void set_cell_double_arrays_to_zero(
        CellDouble *m_c_jx,
        CellDouble *m_c_jy,
        CellDouble *m_c_jz,
        int size,
        int index,
        int blockDimX
) {
    for (int i = 0; i < size; i++) {
        setCellDoubleToZero(&(m_c_jx[i]), index);
        setCellDoubleToZero(&(m_c_jy[i]), index);
        setCellDoubleToZero(&(m_c_jz[i]), index);
    }

    while (index < CellExtent * CellExtent * CellExtent) {
        for (int i = 0; i < size; i++) {}
        index += blockDimX;
    }

    __syncthreads();

}

__device__ void MoveParticlesInCell(
        CellDouble *c_ex,
        CellDouble *c_ey,
        CellDouble *c_ez,
        CellDouble *c_hx,
        CellDouble *c_hy,
        CellDouble *c_hz,
        Cell *c,
        int index,
        int blockDimX//,
) {
    CellTotalField cf;

    while (index < c->number_of_particles) {
        cf.Ex = c->Ex;
        cf.Ey = c->Ey;
        cf.Ez = c->Ez;
        cf.Hx = c->Hx;
        cf.Hy = c->Hy;
        cf.Hz = c->Hz;

        c->MoveSingleParticle(index, cf);


        index += blockDimX;
    }


    __syncthreads();
}

__device__ void AccumulateCurrentWithParticlesInCell(
        CellDouble *c_jx,
        CellDouble *c_jy,
        CellDouble *c_jz,
        Cell *c,
        int index,
        int blockDimX,
        int nt
) {
    DoubleCurrentTensor dt;
    int pqr2;

    while (index < c->number_of_particles) {
        c->AccumulateCurrentSingleParticle(index, &pqr2, &dt);

        writeCurrentComponent(c_jx, &(dt.t1.Jx), &(dt.t2.Jx), pqr2);

        writeCurrentComponent(c_jy, &(dt.t1.Jy), &(dt.t2.Jy), pqr2);
        writeCurrentComponent(c_jz, &(dt.t1.Jz), &(dt.t2.Jz), pqr2);

        index += blockDimX;
    }
    __syncthreads();

}

__device__ void copyFromSharedMemoryToCell(
        CellDouble *c_jx,
        CellDouble *c_jy,
        CellDouble *c_jz,
        Cell *c,
        int index,
        int blockDimX,
        dim3 blockId
) {
    while (index < CellExtent * CellExtent * CellExtent) {
        copyCellDouble(c->Jx, c_jx, index, blockIdx);
        copyCellDouble(c->Jy, c_jy, index, blockIdx);
        copyCellDouble(c->Jz, c_jz, index, blockIdx);

        index += blockDim.x;
    }
    c->busyParticleArray = 0;
}


__global__ void GPU_StepAllCells(GPUCell **cells) {
    Cell *c, *c0 = cells[0];
    __shared__
    CellDouble fd[9];
    CellDouble *c_jx, *c_jy, *c_jz, *c_ex, *c_ey, *c_ez, *c_hx, *c_hy, *c_hz;
    Particle p;

    c = cells[c0->getGlobalCellNumber(blockIdx.x, blockIdx.y, blockIdx.z)];

    assignSharedWithLocal(&c_jx, &c_jy, &c_jz, &c_ex, &c_ey, &c_ez, &c_hx, &c_hy, &c_hz, fd);


    copyFieldsToSharedMemory(c_jx, c_jy, c_jz, c_ex, c_ey, c_ez, c_hx, c_hy, c_hz, c,
                             threadIdx.x, blockIdx, blockDim.x);


    MoveParticlesInCell(c_ex, c_ey, c_ez, c_hx, c_hy, c_hz, c, threadIdx.x, blockDim.x);

    copyFromSharedMemoryToCell(c_jx, c_jy, c_jz, c, threadIdx.x, blockDim.x, blockIdx);
}


__global__ void GPU_CurrentsAllCells(GPUCell **cells, int nt) {
    Cell *c, *c0 = cells[0];
    __shared__
    CellDouble fd[9];
    CellDouble *c_jx, *c_jy, *c_jz, *c_ex, *c_ey, *c_ez, *c_hx, *c_hy, *c_hz;
    __shared__
    CellDouble m_c_jx[CURRENT_SUM_BUFFER_LENGTH];
    __shared__
    CellDouble m_c_jy[CURRENT_SUM_BUFFER_LENGTH];
    __shared__
    CellDouble m_c_jz[CURRENT_SUM_BUFFER_LENGTH];

    c = cells[c0->getGlobalCellNumber(blockIdx.x, blockIdx.y, blockIdx.z)];

    assignSharedWithLocal(&c_jx, &c_jy, &c_jz, &c_ex, &c_ey, &c_ez, &c_hx, &c_hy, &c_hz, fd);


    copyFieldsToSharedMemory(c_jx, c_jy, c_jz, c_ex, c_ey, c_ez, c_hx, c_hy, c_hz, c, threadIdx.x, blockIdx, blockDim.x);

    set_cell_double_arrays_to_zero(m_c_jx, m_c_jy, m_c_jz, CURRENT_SUM_BUFFER_LENGTH, threadIdx.x, blockDim.x);

    AccumulateCurrentWithParticlesInCell(c_jx, c_jy, c_jz, c, threadIdx.x, blockDim.x, nt);

    copyFromSharedMemoryToCell(c_jx, c_jy, c_jz, c, threadIdx.x, blockDim.x, blockIdx);

}

__host__ __device__
void emh2_Element(Cell *c, int i, int l, int k, double *Q, double *H) {
    int n = c->getGlobalCellNumber(i, l, k);

    H[n] += Q[n];
}


__global__
void GPU_emh2(GPUCell **cells, int i_s, int l_s, int k_s, double *Q, double *H) {
    unsigned int nx = blockIdx.x;
    unsigned int ny = blockIdx.y;
    unsigned int nz = blockIdx.z;
    Cell *c0 = cells[0];

    emh2_Element(c0, i_s + nx, l_s + ny, k_s + nz, Q, H);
}


__host__ __device__
void emh1_Element(Cell *c, int3 i, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2) {
    int n = c->getGlobalCellNumber(i.x, i.y, i.z);
    int n1 = c->getGlobalCellNumber(i.x + d1.x, i.y + d1.y, i.z + d1.z);
    int n2 = c->getGlobalCellNumber(i.x + d2.x, i.y + d2.y, i.z + d2.z);

    double e1_n1 = E1[n1];
    double e1_n = E1[n];
    double e2_n2 = E2[n2];
    double e2_n = E2[n];

    double t = 0.5 * (c1 * (e1_n1 - e1_n) - c2 * (e2_n2 - e2_n));
    Q[n] = t;
    H[n] += Q[n];
}

__global__
void GPU_emh1(GPUCell **cells, double *Q, double *H, double *E1, double *E2, double c1, double c2, int3 d1, int3 d2) {
    int3 i3 = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
    Cell *c0 = cells[0];
    emh1_Element(c0, i3, Q, H, E1, E2, c1, c2, d1, d2);
}

__host__ __device__
void emeElement(Cell *c, int3 i, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau, int3 d1, int3 d2) {
    int n = c->getGlobalCellNumber(i.x, i.y, i.z);
    int n1 = c->getGlobalCellNumber(i.x + d1.x, i.y + d1.y, i.z + d1.z);
    int n2 = c->getGlobalCellNumber(i.x + d2.x, i.y + d2.y, i.z + d2.z);

    E[n] += c1 * (H1[n] - H1[n1]) - c2 * (H2[n] - H2[n2]) - tau * J[n];
}

__host__ __device__
void periodicElement(Cell *c, int i, int k, double *E, int dir, int to, int from) {
    int n = c->getGlobalBoundaryCellNumber(i, k, dir, to);
    int n1 = c->getGlobalBoundaryCellNumber(i, k, dir, from);
    E[n] = E[n1];
}

__global__ void GPU_periodic(GPUCell **cells, int i_s, int k_s, double *E, int dir, int to, int from) {
    unsigned int nx = blockIdx.x;
    unsigned int nz = blockIdx.z;
    Cell *c0 = cells[0];

    periodicElement(c0, nx + i_s, nz + k_s, E, dir, to, from);
}

__host__ __device__
void periodicCurrentElement(Cell *c, int i, int k, double *E, int dir, int dirE, int N) {
    int n1 = c->getGlobalBoundaryCellNumber(i, k, dir, 1);
    int n_Nm1 = c->getGlobalBoundaryCellNumber(i, k, dir, N - 1);

    if (dir != dirE) {
        E[n1] += E[n_Nm1];
    }

    if (dir != 1 || dirE != 1) {
        E[n_Nm1] = E[n1];
    }

    int n_Nm2 = c->getGlobalBoundaryCellNumber(i, k, dir, N - 2);
    int n0 = c->getGlobalBoundaryCellNumber(i, k, dir, 0);

#ifdef PERIODIC_CURRENT_PRINTS
    printf("%e %e \n",E[n0],E[n_Nm2]);
#endif
    E[n0] += E[n_Nm2];
    E[n_Nm2] = E[n0];
}


__global__ void GPU_CurrentPeriodic(GPUCell **cells, double *E, int dirE, int dir, int i_s, int k_s, int N) {
    unsigned int nx = blockIdx.x;
    unsigned int nz = blockIdx.z;
    Cell *c0 = cells[0];
    periodicCurrentElement(c0, nx + i_s, nz + k_s, E, dir, dirE, N);
}


__global__ void GPU_eme(GPUCell **cells, int3 s, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau, int3 d1, int3 d2) {
    unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int nz = blockIdx.z * blockDim.z + threadIdx.z;
    Cell *c0 = cells[0];

    s.x += nx;
    s.y += ny;
    s.z += nz;

    emeElement(c0, s, E, H1, H2, J, c1, c2, tau, d1, d2);
}
