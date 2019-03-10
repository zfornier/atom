//
// Created by egor on 19.02.19.
//

#include "../include/Plasma.h"

Plasma::Plasma(int nx, int ny, int nz, double lx, double ly, double lz, double ni1, int n_per_cell1, double q_m,
               double TAU) {
    Nx = nx;
    Ny = ny;
    Nz = nz;

    Lx = lx;
    Ly = ly;
    Lz = lz;

    ni = ni1;

    n_per_cell = n_per_cell1;
    ion_q_m = q_m;
    tau = TAU;
    dataFileStartPattern = "data";
    dataFileEndPattern = ".nc";
}

Plasma::~Plasma() {}

void Plasma::copyCells(std::string where, int nt) {
    static int first = 1;
    size_t m_free, m_total;
    int size = (*AllCells).size();
    struct sysinfo info;

    if (first == 1) {
        cp = (GPUCell **) malloc(size * sizeof(GPUCell *));
    }

    unsigned long m1, m2, delta, accum;
    memory_monitor("beforeCopyCells", nt);

    for (int i = 0; i < size; i++) {
        cudaError_t err = cudaMemGetInfo(&m_free, &m_total);
        sysinfo(&info);
        m1 = info.freeram;
        GPUCell c, *d_c, *z0;
        z0 = h_CellArray[i];
        if (first == 1) {
            d_c = c.allocateCopyCellFromDevice();
            cp[i] = d_c;
        } else {
            d_c = cp[i];
        }
        c.copyCellFromDevice(z0, d_c, where, nt);
        m2 = info.freeram;

        delta = m1 - m2;
        accum += delta;

    }

    if (first == 1) {
        first = 0;
    }

    memory_monitor("afterCopyCells", nt);
}

double Plasma::checkGPUArray(double *a, double *d_a, std::string name, std::string where, int nt) {
    static double *t;
    static int f1 = 1;
    char fname[1000];
    double res;

#ifndef CHECK_ARRAY_OUTPUT
    return 0.0;
#endif

    sprintf(fname, "diff_%s_at_%s_nt%08d.dat", name.c_str(), where.c_str(), nt);


    if (f1 == 1) {
        t = (double *) malloc(sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2));
        f1 = 0;
    }
    int err;
    err = MemoryCopy(t, d_a, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    if (err != cudaSuccess) {
        printf("bCheckArray err %d %s \n", err, getErrorString(err));
        exit(0);
    }

#ifdef CHECK_ARRAY_DETAIL_PRINTS
    if((f = fopen(fname,"wt")) != NULL) {
        res = CheckArray(a,t,f);
        fclose(f);
    }
#else
    int size = (Nx + 2) * (Ny + 2) * (Nz + 2);
    res = CheckArraySilent(a, t, size);
#endif

    return res;
}

double Plasma::checkGPUArray(double *a, double *d_a) {
    static double *t;
    static int f = 1;

    if (f == 1) {
        t = (double *) malloc(sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2));
        f = 0;
    }
    int err;
    err = MemoryCopy(t, d_a, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    if (err != cudaSuccess) {
        printf("bCheckArray err %d %s \n", err, getErrorString(err));
        exit(0);
    }

    return CheckArray(a, t);
}

void
Plasma::emeGPUIterate(int3 s, int3 f, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau,
                      int3 d1, int3 d2) {
    dim3 dimGrid(f.x - s.x + 1, 1, 1), dimBlock(1, f.y - s.y + 1, f.z - s.z + 1);

    void *args[] = {(void *) &d_CellArray,
                    (void *) &s,
                    (void *) &E,
                    (void *) &H1,
                    (void *) &H2,
                    (void *) &J,
                    (void *) &c1,
                    (void *) &c2,
                    (void *) &tau,
                    (void *) &d1,
                    (void *) &d2,
                    0};

    cudaError_t cudaStatus = cudaLaunchKernel(
            (const void *) GPU_eme,        // pointer to kernel func.
            dimGrid,                       // grid
            dimBlock,                      // block
            args,                          // arguments
            0,
            0
    );
}

void Plasma::GetElectricFieldStartsDirs(int3 *start, int3 *d1, int3 *d2, int dir) {
    start->x = (dir == 0) * 0 + (dir == 1) * 1 + (dir == 2) * 1;
    start->y = (dir == 0) * 1 + (dir == 1) * 0 + (dir == 2) * 1;
    start->z = (dir == 0) * 1 + (dir == 1) * 1 + (dir == 2) * 0;

    d1->x = (dir == 0) * 0 + (dir == 1) * 0 + (dir == 2) * (-1);
    d1->y = (dir == 0) * (-1) + (dir == 1) * 0 + (dir == 2) * 0;
    d1->z = (dir == 0) * 0 + (dir == 1) * (-1) + (dir == 2) * 0;

    d2->x = (dir == 0) * 0 + (dir == 1) * (-1) + (dir == 2) * 0;
    d2->y = (dir == 0) * 0 + (dir == 1) * 0 + (dir == 2) * (-1);
    d2->z = (dir == 0) * (-1) + (dir == 1) * 0 + (dir == 2) * 0;
}

int
Plasma::ElectricFieldTrace(double *E, double *H1, double *H2, double *J, int dir, double c1, double c2, double tau) {
    int3 start, d1, d2, finish = make_int3(Nx, Ny, Nz);

    GetElectricFieldStartsDirs(&start, &d1, &d2, dir);

    emeGPUIterate(start, finish, E, H1, H2, J, c1, c2, tau, d1, d2);

    return 0;
}

int Plasma::checkFields_beforeMagneticStageOne(double *t_Ex, double *t_Ey, double *t_Ez,
                                               double *t_Hx, double *t_Hy, double *t_Hz,
                                               double *t_Qx, double *t_Qy, double *t_Qz,
                                               double *t_check, int nt) {

    memory_monitor("beforeComputeField_FirstHalfStep", nt);

    return 0;
}

int Plasma::checkFields_afterMagneticStageOne(double *t_Hx, double *t_Hy, double *t_Hz, double *t_Qx, double *t_Qy,
                                              double *t_Qz, double *t_check, int nt) {

    CPU_field = 1;

    checkControlPoint(50, nt, 0);
    memory_monitor("afterComputeField_FirstHalfStep", nt);

    return 0;
}

void Plasma::ComputeField_FirstHalfStep(int nt) {
    double t_check[15];
    cudaError_t err;

    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }

    checkFields_beforeMagneticStageOne(d_Ex, d_Ey, d_Ez, d_Hx, d_Hy, d_Hz, d_Qx, d_Qy, d_Qz, t_check, nt);
    err = cudaGetLastError();

    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }
    MagneticStageOne(d_Qx, d_Qy, d_Qz, d_Hx, d_Hy, d_Hz, d_Ex, d_Ey, d_Ez);

    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }

    checkFields_afterMagneticStageOne(d_Hx, d_Hy, d_Hz, d_Qx, d_Qy, d_Qz, t_check, nt);

    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }

    AssignCellsToArraysGPU();

    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }
}

void Plasma::ComputeField_SecondHalfStep(int nt) {

    SetPeriodicCurrents(nt);


    MagneticFieldStageTwo(d_Hx, d_Hy, d_Hz, nt, d_Qx, d_Qy, d_Qz);


    ElectricFieldEvaluate(d_Ex, d_Ey, d_Ez, nt, d_Hx, d_Hy, d_Hz, d_Jx, d_Jy, d_Jz);


}

void Plasma::ElectricFieldComponentEvaluateTrace(
        double *E, double *H1, double *H2, double *J,
        int dir, double c1, double c2, double tau,
        int dir_1, int start1_1, int end1_1, int start2_1, int end2_1, int N_1,
        int dir_2, int start1_2, int end1_2, int start2_2, int end2_2, int N_2
) {
    ElectricFieldTrace(E, H1, H2, J, dir, c1, c2, tau);
}

void Plasma::ElectricFieldComponentEvaluatePeriodic(
        double *E, double *H1, double *H2, double *J,
        int dir, double c1, double c2, double tau,
        int dir_1, int start1_1, int end1_1, int start2_1, int end2_1, int N_1,
        int dir_2, int start1_2, int end1_2, int start2_2, int end2_2, int N_2
) {
    if (dir != 0) {
        PeriodicBoundaries(E, dir_1, start1_1, end1_1, start2_1, end2_1, N_1);
    }

    if (dir != 2) {
        PeriodicBoundaries(E, dir_2, start1_2, end1_2, start2_2, end2_2, N_2);
    }
}

void Plasma::ElectricFieldEvaluate(double *locEx, double *locEy, double *locEz,
                                   int nt,
                                   double *locHx, double *locHy, double *locHz,
                                   double *loc_npJx, double *loc_npJy, double *loc_npJz) {
    CPU_field = 0;
    double3 c1 = getMagneticFieldTimeMeshFactors();

    ElectricFieldComponentEvaluateTrace(
            locEx, locHz, locHy, loc_npJx,
            0, c1.y, c1.z, tau,
            1, 0, Nx, 1, Nz, Ny,
            2, 0, Nx, 0, Ny + 1, Nz);

    ElectricFieldComponentEvaluateTrace(
            locEy, locHx, locHz, loc_npJy,
            1, c1.z, c1.x, tau,
            0, 0, Ny, 1, Nz, Nx,
            2, 0, Nx + 1, 0, Ny, Nz);

    ElectricFieldComponentEvaluateTrace(
            locEz, locHy, locHx, loc_npJz,
            2, c1.x, c1.y, tau,
            0, 1, Ny, 0, Nz, Nx,
            1, 0, Nx + 1, 0, Nz, Ny);

    checkControlPoint(550, nt, 0);

    ElectricFieldComponentEvaluatePeriodic(
            locEx, locHz, locHy, loc_npJx,
            0, c1.y, c1.z, tau,
            1, 0, Nx, 1, Nz, Ny,
            2, 0, Nx, 0, Ny + 1, Nz);

    ElectricFieldComponentEvaluatePeriodic(
            locEy, locHx, locHz, loc_npJy,
            1, c1.z, c1.x, tau,
            0, 0, Ny, 1, Nz, Nx,
            2, 0, Nx + 1, 0, Ny, Nz);

    ElectricFieldComponentEvaluatePeriodic(
            locEz, locHy, locHx, loc_npJz,
            2, c1.x, c1.y, tau,
            0, 1, Ny, 0, Nz, Nx,
            1, 0, Nx + 1, 0, Nz, Ny);

    checkControlPoint(600, nt, 0);

    memory_monitor("after_ComputeField_SecondHalfStep", nt);
}

double3 Plasma::getMagneticFieldTimeMeshFactors() {
    Cell c = (*AllCells)[0];
    double hx = c.get_hx(), hy = c.get_hy(), hz = c.get_hz();
    double3 d;
    d.x = tau / (hx);
    d.y = tau / (hy);
    d.z = tau / hz;

    return d;
}

void Plasma::MagneticStageOne(
        double *Qx, double *Qy, double *Qz,
        double *Hx, double *Hy, double *Hz,
        double *Ex, double *Ey, double *Ez
) {
    double3 c1 = getMagneticFieldTimeMeshFactors();

    MagneticFieldTrace(Qx, Hx, Ey, Ez, Nx + 1, Ny, Nz, c1.z, c1.y, 0);
    MagneticFieldTrace(Qy, Hy, Ez, Ex, Nx, Ny + 1, Nz, c1.x, c1.z, 1);
    MagneticFieldTrace(Qz, Hz, Ex, Ey, Nx, Ny, Nz + 1, c1.y, c1.x, 2);
}

void Plasma::MagneticFieldStageTwo(double *Hx, double *Hy, double *Hz,
                                   int nt,
                                   double *Qx, double *Qy, double *Qz) {
    Cell c = (*AllCells)[0];

    SimpleMagneticFieldTrace(c, Qx, Hx, Nx + 1, Ny, Nz);
    SimpleMagneticFieldTrace(c, Qy, Hy, Nx, Ny + 1, Nz);
    SimpleMagneticFieldTrace(c, Qz, Hz, Nx, Ny, Nz + 1);

    checkControlPoint(500, nt, 0);
}

int Plasma::PushParticles(int nt) {
    double mass = -1.0 / 1836.0, q_mass = -1.0;

    memory_monitor("before_CellOrder_StepAllCells", nt);

    CellOrder_StepAllCells(nt, mass, q_mass, 1);
    puts("cell_order");

    memory_monitor("after_CellOrder_StepAllCells", nt);

    checkControlPoint(270, nt, 1);

    return 0;
}

int Plasma::readStartPoint(int nt) {
    char fn[100];

    if (nt == START_STEP_NUMBER) {
        readControlPoint(NULL, fn, 0, nt, 0, 1, Ex, Ey, Ez, Hx, Hy, Hz, Jx, Jy, Jz, Qx, Qy, Qz,
                         dbg_x, dbg_y, dbg_z, dbg_px, dbg_py, dbg_pz
        );

        copyFieldsToGPU(
                d_Ex, d_Ey, d_Ez,
                d_Hx, d_Hy, d_Hz,
                d_Jx, d_Jy, d_Jz,
                d_npJx, d_npJy, d_npJz,
                d_Qx, d_Qy, d_Qz,
                Ex, Ey, Ez,
                Hx, Hy, Hz,
                Jx, Jy, Jz,
                npJx, npJy, npJz,
                Qx, Qy, Qz,
                Nx, Ny, Nz
        );
    }

    checkControlPoint(0, nt, 1);

    return 0;
}

double Plasma::getElectricEnergy() {
    dim3 dimGrid(Nx + 2, Ny + 2, Nz + 2), dimGridOne(1, 1, 1), dimBlock(MAX_particles_per_cell / 2, 1, 1),
            dimBlockOne(1, 1, 1), dimBlockGrow(1, 1, 1), dimBlockExt(CellExtent, CellExtent, CellExtent);
    static int first = 1;
    static double *d_ee;
    double ee;

    if (first == 1) {
        cudaMalloc((void **) &d_ee, sizeof(double));
        first = 0;
    }
    cudaMemset(d_ee, 0, sizeof(double));

    void *args[] = {(void *) &d_CellArray,
                    (void *) &d_ee,
                    (void *) &d_Ex,
                    (void *) &d_Ey,
                    (void *) &d_Ez,
                    0};
    cudaError_t cudaStatus = cudaLaunchKernel(
            (const void *) GPU_getCellEnergy, // pointer to kernel func.
            dimGrid,                          // grid
            dimBlockOne,                      // block
            args,                             // arguments
            0,
            0
    );


    MemoryCopy(&ee, d_ee, sizeof(double), DEVICE_TO_HOST);

    return ee;
}

void Plasma::Diagnose(int nt) {
    double ee;
    static FILE *f;
    static int first = 1;

    if (first == 1) {
        f = fopen("energy.dat", "wt");
        first = 0;
    } else {
        f = fopen("energy.dat", "at");

    }

    ee = getElectricEnergy();
    // sumMPIenergy(&ee);

    fprintf(f, "%10d %25.15e \n", nt, ee);

    fclose(f);
}

int Plasma::getBoundaryLimit(int dir) { return ((dir == 0) * Nx + (dir == 1) * Ny + (dir == 2) * Nz + 2); }

int Plasma::getMagneticFieldTraceShifts(int dir, int3 &d1, int3 &d2) {
    d1.x = (dir == 0) * 0 + (dir == 1) * 1 + (dir == 2) * 0;
    d1.y = (dir == 0) * 0 + (dir == 1) * 0 + (dir == 2) * 1;
    d1.z = (dir == 0) * 1 + (dir == 1) * 0 + (dir == 2) * 0;

    d2.x = (dir == 0) * 0 + (dir == 1) * 0 + (dir == 2) * 1;
    d2.y = (dir == 0) * 1 + (dir == 1) * 0 + (dir == 2) * 0;
    d2.z = (dir == 0) * 0 + (dir == 1) * 1 + (dir == 2) * 0;

    return 0;
}

int Plasma::MagneticFieldTrace(double *Q, double *H, double *E1, double *E2, int i_end, int l_end, int k_end, double c1,
                               double c2, int dir) {
    int3 d1, d2;

    getMagneticFieldTraceShifts(dir, d1, d2);

    dim3 dimGrid(i_end + 1, l_end + 1, k_end + 1), dimBlock(1, 1, 1);

    void *args[] = {(void *) &d_CellArray,
                    (void *) &Q,
                    (void *) &H,
                    (void *) &E1,
                    (void *) &E2,
                    (void *) &c1,
                    (void *) &c2,
                    (void *) &d1,
                    (void *) &d2,
                    0};
    cudaError_t cudaStatus = cudaLaunchKernel(
            (const void *) GPU_emh1, // pointer to kernel func.
            dimGrid,                 // grid
            dimBlock,                // block
            args,                    // arguments
            0,
            0
    );

    return 0;
}

int Plasma::SimpleMagneticFieldTrace(Cell &c, double *Q, double *H, int i_end, int l_end, int k_end) {


    dim3 dimGrid(i_end + 1, l_end + 1, k_end + 1), dimBlock(1, 1, 1);

    int i_s = 0;
    int l_s = 0;
    int k_s = 0;

    void *args[] = {(void *) &d_CellArray,
                    (void *) &i_s,
                    (void *) &l_s,
                    (void *) &k_s,
                    (void *) &Q,
                    (void *) &H,
                    0};
    cudaError_t cudaStatus = cudaLaunchKernel(
            (const void *) GPU_emh2,       // pointer to kernel func.
            dimGrid,                       // grid
            dimBlock,                      // block
            args,                          // arguments
            0,
            0
    );

    return 0;
}

int Plasma::PeriodicBoundaries(double *E, int dir, int start1, int end1, int start2, int end2, int N) {
    Cell c = (*AllCells)[0];

    dim3 dimGrid(end1 - start1 + 1, 1, end2 - start2 + 1), dimBlock(1, 1, 1);

    int zero = 0;
    void *args[] = {(void *) &d_CellArray,
                    (void *) &start1,
                    (void *) &start2,
                    (void *) &E,
                    (void *) &dir,
                    (void *) &zero,
                    (void *) &N,
                    0};
    cudaError_t cudaStatus = cudaLaunchKernel(
            (const void *) GPU_periodic,   // pointer to kernel func.
            dimGrid,                       // grid
            dimBlock,                      // block
            args,                          // arguments
            0,
            0
    );

    int one = 1;
    int N1 = N + 1;

    void *args1[] = {(void *) &d_CellArray,
                     (void *) &start1,
                     (void *) &start2,
                     (void *) &E,
                     (void *) &dir,
                     (void *) &N1,
                     (void *) &one,
                     0};
    cudaStatus = cudaLaunchKernel(
            (const void *) GPU_periodic,   // pointer to kernel func.
            dimGrid,                       // grid
            dimBlock,                      // block
            args1,                         // arguments
            0,
            0
    );

    return 0;
}

int Plasma::SinglePeriodicBoundary(double *E, int dir, int start1, int end1, int start2, int end2, int N) {
    Cell c = (*AllCells)[0];

    if (CPU_field == 0) {
        dim3 dimGrid(end1 - start1 + 1, 1, end2 - start2 + 1), dimBlock(1, 1, 1);

        int N1 = N + 1;
        int one = 1;
        void *args[] = {(void *) &d_CellArray,
                        (void *) &start1,
                        (void *) &start2,
                        (void *) &E,
                        (void *) &dir,
                        (void *) &N1,
                        (void *) &one,
                        0};
        cudaError_t cudaStatus = cudaLaunchKernel(
                (const void *) GPU_periodic,   // pointer to kernel func.
                dimGrid,                       // grid
                dimBlock,                      // block
                args,                          // arguments
                16000,
                0
        );


    } else {
        for (int k = start2; k <= end2; k++) {
            for (int i = start1; i <= end1; i++) {
                int3 i0, i1;

                int n = c.getGlobalBoundaryCellNumber(i, k, dir, N + 1);
                int n1 = c.getGlobalBoundaryCellNumber(i, k, dir, 1);
                E[n] = E[n1];
                i0 = c.getCellTripletNumber(n);
                i1 = c.getCellTripletNumber(n1);
                std::cout << "ex1 " << i0.x + 1 << " " << i0.y + 1 << " " << i0.z + 1 << " " << i1.x + 1 << " "
                          << i1.y + 1 << " " << i1.z + 1 << " " << E[n] << " " << E[n1] << std::endl;
            }
        }
    }

    return 0;
}

int Plasma::SetPeriodicCurrentComponent(GPUCell **cells, double *J, int dir, int Nx, int Ny, int Nz) {
    dim3 dimGridX(Ny + 2, 1, Nz + 2), dimGridY(Nx + 2, 1, Nz + 2), dimGridZ(Nx + 2, 1, Ny + 2), dimBlock(1, 1, 1);

    int dir2 = 0;
    int i_s = 0;
    int k_s = 0;
    int N = Nx + 2;
    void *args[] = {(void *) &cells,
                    (void *) &J,
                    (void *) &dir,
                    (void *) &dir2,
                    (void *) &i_s,
                    (void *) &k_s,
                    (void *) &N,
                    0};
    cudaError_t cudaStatus = cudaLaunchKernel(
            (const void *) GPU_CurrentPeriodic, // pointer to kernel func.
            dimGridX,                           // grid
            dimBlock,                           // block
            args,                               // arguments
            16000,
            0
    );

    dir2 = 1;
    N = Ny + 2;
    cudaStatus = cudaLaunchKernel(
            (const void *) GPU_CurrentPeriodic, // pointer to kernel func.
            dimGridY,                           // grid
            dimBlock,                           // block
            args,                               // arguments
            16000,
            0
    );

    dir2 = 2;
    N = Nz + 2;
    cudaStatus = cudaLaunchKernel(
            (const void *) GPU_CurrentPeriodic, // pointer to kernel func.
            dimGridZ,                       // grid
            dimBlock,                      // block
            args,                          // arguments
            16000,
            0
    );

    return 0;
}

void Plasma::SetPeriodicCurrents(int nt) {
    memory_monitor("before275", nt);

    checkControlPoint(275, nt, 0);
    SetPeriodicCurrentComponent(d_CellArray, d_Jx, 0, Nx, Ny, Nz);
    SetPeriodicCurrentComponent(d_CellArray, d_Jy, 1, Nx, Ny, Nz);
    SetPeriodicCurrentComponent(d_CellArray, d_Jz, 2, Nx, Ny, Nz);

    checkControlPoint(400, nt, 0);
}

void Plasma::InitQdebug(std::string fnjx, std::string fnjy, std::string fnjz) {
    read3Darray(fnjx, dbg_Qx);
    read3Darray(fnjy, dbg_Qy);
    read3Darray(fnjz, dbg_Qz);
}

void Plasma::AssignCellsToArraysGPU() {
    dim3 dimGrid(Nx, Ny, Nz), dimBlockExt(CellExtent, CellExtent, CellExtent);
    cudaError_t err = cudaGetLastError();
    printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err));

    size_t sz;
    err = cudaDeviceGetLimit(&sz, cudaLimitStackSize);
    printf("%s:%d - stack limit %d err = %d\n", __FILE__, __LINE__, ((int) sz), err);
    err = cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);
    printf("%s:%d - set stack limit %d \n", __FILE__, __LINE__, err);
    err = cudaDeviceGetLimit(&sz, cudaLimitStackSize);
    printf("%s:%d - stack limit %d err %d\n", __FILE__, __LINE__, ((int) sz), err);

    void *args[] = {(void *) &d_CellArray, &d_Ex, &d_Ey, &d_Ez, &d_Hx, &d_Hy, &d_Hz, 0};
    cudaError_t cudaStatus = cudaLaunchKernel(
            (const void *) GPU_SetFieldsToCells, // pointer to kernel func.
            dimGrid,                             // grid
            dimBlockExt,                         // block
            args,                                // arguments
            16000,
            0
    );

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err));
}

void Plasma::AssignCellsToArrays() {
    for (int n = 0; n < (*AllCells).size(); n++) {
        Cell c = (*AllCells)[n];
        c.writeAllToArrays(Jx, Jy, Jz, Rho, 0);
    }
    CheckArray(Jx, dbgJx);
    SetPeriodicCurrents(0);
    CheckArray(Jx, dbgJx);
}

void Plasma::write3Darray(char *name, double *d) {
    char fname[100];
    GPUCell c = (*AllCells)[0];
    FILE *f;

    sprintf(fname, "%s_fiel3d.dat", name);

    if ((f = fopen(fname, "wt")) == NULL) return;

    for (int i = 1; i < Nx + 1; i++) {
        for (int l = 1; l < Ny + 1; l++) {
            for (int k = 1; k < Nz + 1; k++) {
                int n = c.getGlobalCellNumber(i, l, k);

                fprintf(f, "%15.5e %15.5e %15.5e %25.15e \n", c.getNodeX(i), c.getNodeY(l), c.getNodeZ(k), d[n]);
            }
        }
    }

    fclose(f);
}

void Plasma::write3D_GPUArray(char *name, double *d_d) {
    return;
}

void Plasma::readControlPoint(FILE **f1, char *fncpy, int num, int nt, int part_read, int field_assign,
                              double *ex, double *ey, double *ez,
                              double *hx, double *hy, double *hz,
                              double *jx, double *jy, double *jz,
                              double *qx, double *qy, double *qz,
                              double *x, double *y, double *z,
                              double *px, double *py, double *pz
) {
    char fn[100], fn_next[100];
    FILE *f;

    sprintf(fn, "mumu%03d%08d.nc", num, nt);
    strcpy(fncpy, fn);
    sprintf(fn_next, "mumu%03d%05d.nc", num, nt + 1);
    if ((f = fopen(fn, "rb")) == NULL) {
        std::cerr << "Error: file " << fn << " not found" << std::endl;
        return;
    }
    if (part_read) {
        *f1 = f;
    }

    std::cout << fn << std::endl;
    readVar(fn, "Ex", (void *) ex);
    readVar(fn, "Ey", (void *) ey);
    readVar(fn, "Ez", (void *) ez);

    readVar(fn, "Mx", (void *) hx);
    readVar(fn, "My", (void *) hy);
    readVar(fn, "Mz", (void *) hz);

    readVar(fn, "Jx", (void *) jx);
    readVar(fn, "Jy", (void *) jy);
    readVar(fn, "Jz", (void *) jz);

    readVar(fn, "Qx", (void *) qx);
    readVar(fn, "Qy", (void *) qy);
    readVar(fn, "Qz", (void *) qz);

//	readFortranBinaryArray(f,ex);
//	readFortranBinaryArray(f,ey);
//	readFortranBinaryArray(f,ez);
//	readFortranBinaryArray(f,hx);
//	readFortranBinaryArray(f,hy);
//	readFortranBinaryArray(f,hz);
//	readFortranBinaryArray(f,jx);
//	readFortranBinaryArray(f,jy);
//	readFortranBinaryArray(f,jz);

//	readFortranBinaryArray(f,qx);
//	readFortranBinaryArray(f,qy);
//	readFortranBinaryArray(f,qz);

    if (field_assign == 1) AssignArraysToCells();
}

double Plasma::checkControlMatrix(char *wh, int nt, char *name, double *d_m) {
    double t_jx; //,t_jy;//,t_jz;
    char fn[100];
    FILE *f;

#ifndef CHECK_CONTROL_MATRIX
    return 0.0;
#endif

    sprintf(fn, "wcmx_%4s_%08d_%2s.dat", wh, nt, name);
    if ((f = fopen(fn, "rb")) == NULL) return -1.0;

    readFortranBinaryArray(f, dbgJx);

    t_jx = checkGPUArray(dbgJx, d_m);

    return t_jx;
}

void Plasma::checkCurrentControlPoint(int j, int nt) {
    double /*t_ex,t_ey,t_ez,t_hx,t_hy,t_hz,*/t_jx, t_jy, t_jz;
    char fn[100];//,fn_next[100];
    FILE *f;

    sprintf(fn, "curr%03d%05d.dat", nt, j);
    if ((f = fopen(fn, "rb")) == NULL) return;

    readFortranBinaryArray(f, dbgJx);
    readFortranBinaryArray(f, dbgJy);
    readFortranBinaryArray(f, dbgJz);

    int size = (Nx + 2) * (Ny + 2) * (Nz + 2);

    t_jx = CheckArraySilent(Jx, dbgJx, size);
    t_jy = CheckArraySilent(Jy, dbgJy, size);
    t_jz = CheckArraySilent(Jz, dbgJz, size);

    printf("Jx %15.5e Jy %15.5e Jz %15.5e \n", t_jx, t_jy, t_jz);
}

void Plasma::checkControlPoint(int num, int nt, int check_part) {
    double t_ex, t_ey, t_ez, t_hx, t_hy, t_hz, t_jx, t_jy, t_jz;
    double t_qx, t_qy, t_qz;//,t_njx,t_njy,t_njz;

    if ((nt != TOTAL_STEPS) || (num != 600)) {
#ifndef CONTROL_POINT_CHECK
        return;
#endif
    }

    FILE *f;
    char fn_copy[100];

    memory_monitor("checkControlPoint1", nt);

    if (nt % FORTRAN_NUMBER_OF_SMALL_STEPS != 0) return;

    memory_monitor("checkControlPoint2", nt);

    readControlPoint(&f, fn_copy, num, nt, 1, 0, dbgEx, dbgEy, dbgEz, dbgHx, dbgHy, dbgHz, dbgJx, dbgJy, dbgJz,
                     dbg_Qx, dbg_Qy, dbg_Qz,
                     dbg_x, dbg_y, dbg_z, dbg_px, dbg_py, dbg_pz);

    memory_monitor("checkControlPoint3", nt);

    int size = (Nx + 2) * (Ny + 2) * (Nz + 2);


    t_ex = CheckArraySilent(Ex, dbgEx, size);
    t_ey = CheckArraySilent(Ey, dbgEy, size);
    t_ez = CheckArraySilent(Ez, dbgEz, size);
    t_hx = CheckArraySilent(Hx, dbgHx, size);
    t_hy = CheckArraySilent(Hy, dbgHy, size);
    t_hz = CheckArraySilent(Hz, dbgHz, size);
    t_jx = CheckArraySilent(Jx, dbgJx, size);
    t_jy = CheckArraySilent(Jy, dbgJy, size);
    t_jz = CheckArraySilent(Jz, dbgJz, size);

    memory_monitor("checkControlPoint4", nt);

    t_ex = CheckGPUArraySilent(dbgEx, d_Ex);
    t_ey = CheckGPUArraySilent(dbgEy, d_Ey);
    t_ez = CheckGPUArraySilent(dbgEz, d_Ez);
    t_hx = CheckGPUArraySilent(dbgHx, d_Hx);
    t_hy = CheckGPUArraySilent(dbgHy, d_Hy);
    t_hz = CheckGPUArraySilent(dbgHz, d_Hz);

    t_qx = CheckGPUArraySilent(dbg_Qx, d_Qx);
    t_qy = CheckGPUArraySilent(dbg_Qy, d_Qy);
    t_qz = CheckGPUArraySilent(dbg_Qz, d_Qz);

    if (num >= 500) {
        char wh[100];

        sprintf(wh, "%d", num);

        t_jx = checkGPUArray(dbgJx, d_Jx, "Jx", wh, nt); //checkGPUArrayСomponent(dbgEx,d_Ex,"Ex",num);
        t_jy = checkGPUArray(dbgJy, d_Jy, "Jy", wh, nt);
        t_jz = checkGPUArray(dbgJz, d_Jz, "Jz", wh, nt);

        t_ex = checkGPUArray(dbgEx, d_Ex, "Ex", wh, nt); //checkGPUArrayСomponent(dbgEx,d_Ex,"Ex",num);
        t_ey = checkGPUArray(dbgEy, d_Ey, "Ey", wh, nt);
        t_ez = checkGPUArray(dbgEz, d_Ez, "Ez", wh, nt);

    } else {

        t_ex = CheckGPUArraySilent(dbgEx, d_Ex);
        t_ey = CheckGPUArraySilent(dbgEy, d_Ey);
        t_ez = CheckGPUArraySilent(dbgEz, d_Ez);
    }

    t_jx = CheckGPUArraySilent(dbgJx, d_Jx);
    t_jy = CheckGPUArraySilent(dbgJy, d_Jy);
    t_jz = CheckGPUArraySilent(dbgJz, d_Jz);

    memory_monitor("checkControlPoint5", nt);

    double t_cmp_jx = checkGPUArray(dbgJx, d_Jx, "Jx", "step", nt);
    double t_cmp_jy = checkGPUArray(dbgJy, d_Jy, "Jy", "step", nt);
    double t_cmp_jz = checkGPUArray(dbgJz, d_Jz, "Jz", "step", nt);

#ifdef CONTROL_DIFF_GPU_PRINTS
    printf("GPU: Ex %15.5e Ey %15.5e Ez %15.5e \n",t_ex,t_ey,t_ez);
printf("GPU: Hx %15.5e Hy %15.5e Ez %15.5e \n",t_hx,t_hy,t_hz);
printf("GPU: Jx %15.5e Jy %15.5e Jz %15.5e \n",t_jx,t_jy,t_jz);
printf("GPU compare : Jx %15.5e Jy %15.5e Jz %15.5e \n",t_cmp_jx,t_cmp_jy,t_cmp_jz);
#endif

    memory_monitor("checkControlPoint6", nt);

    double cp = checkControlPointParticles(num, f, fn_copy, nt);
    printf("STEP: %d\n", nt);
    f_prec_report = fopen("control_points.dat", "at");
    fprintf(f_prec_report,
            "nt %5d num %3d Ex %15.5e Ey %15.5e Ez %15.5e Hx %15.5e Hy %15.5e Hz %15.5e Jx %15.5e Jy %15.5e Jz %15.5e Qx %15.5e Qy %15.5e Qz %15.5e particles %15.5e\n",
            nt, num,
            t_ex, t_ey, t_ez,
            t_hx, t_hy, t_hz,
            t_jx, t_jy, t_jz,
            t_qx, t_qy, t_qz,
            cp
    );

    fclose(f_prec_report);

    memory_monitor("checkControlPoint7", nt);

    fclose(f);
}

void Plasma::copyCellCurrentsToDevice(CellDouble *d_jx, CellDouble *d_jy, CellDouble *d_jz, CellDouble *h_jx,
                                      CellDouble *h_jy, CellDouble *h_jz) {
    int err;

    err = MemoryCopy(d_jx, h_jx, sizeof(CellDouble), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("1copyCellCurrentsToDevice err %d %s \n", err, getErrorString(err));
        exit(0);
    }
    err = MemoryCopy(d_jy, h_jy, sizeof(CellDouble), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("2copyCellCurrentsToDevice err %d %s \n", err, getErrorString(err));
        exit(0);
    }
    err = MemoryCopy(d_jz, h_jz, sizeof(CellDouble), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("3copyCellCurrentsToDevice err %d %s \n", err, getErrorString(err));
        exit(0);
    }
}

double Plasma::CheckArray(double *a, double *dbg_a, FILE *f) {
    Cell c = (*AllCells)[0];
    double diff = 0.0;


#ifdef CHECK_ARRAY_DETAIL_PRINTS
    fprintf(f,"begin array checking=============================\n");
#endif
    for (int n = 0; n < (Nx + 2) * (Ny + 2) * (Nz + 2); n++) {
        diff += pow(a[n] - dbg_a[n], 2.0);

        if (fabs(a[n] - dbg_a[n]) > TOLERANCE) {

            int3 i = c.getCellTripletNumber(n);
#ifdef CHECK_ARRAY_DETAIL_PRINTS
            fprintf(f,"n %5d i %3d l %3d k %3d %15.5e dbg %15.5e diff %15.5e wrong %10d \n", n,i.x+1,i.y+1,i.z+1,a[n],dbg_a[n],fabs(a[n] - dbg_a[n]),wrong++);
#endif
        }
    }
#ifdef CHECK_ARRAY_DETAIL_PRINTS
    fprintf(f,"  end array checking============================= %.4f less than %15.5e diff %15.5e \n",
(1.0-((double)wrong/((Nx + 2)*(Ny + 2)*(Nz + 2)))),TOLERANCE,
pow(diff/((Nx + 2)*(Ny + 2)*(Nz + 2)),0.5)
);
#endif

    return pow(diff, 0.5);
}

double Plasma::CheckArray(double *a, double *dbg_a) {
    Cell c = (*AllCells)[0];
    int wrong = 0;
    double diff = 0.0;
#ifdef CHECK_ARRAY_DETAIL_PRINTS
    puts("begin array checking2=============================");
#endif
    for (int n = 0; n < (Nx + 2) * (Ny + 2) * (Nz + 2); n++) {
        diff += pow(a[n] - dbg_a[n], 2.0);

        if (fabs(a[n] - dbg_a[n]) > TOLERANCE) {

            int3 i = c.getCellTripletNumber(n);
#ifdef CHECK_ARRAY_DETAIL_PRINTS
            printf("n %5d i %3d l %3d k %3d %15.5e dbg %15.5e diff %15.5e wrong %10d \n", n,i.x+1,i.y+1,i.z+1,a[n],dbg_a[n],fabs(a[n] - dbg_a[n]),wrong++);
#endif
        }
    }
#ifdef CHECK_ARRAY_DETAIL_PRINTS
    printf("  end array checking============================= %.4f less than %15.5e diff %15.5e \n", (1.0-((double)wrong/((Nx + 2)*(Ny + 2)*(Nz + 2)))),TOLERANCE, pow(diff/((Nx + 2)*(Ny + 2)*(Nz + 2)),0.5));
#endif

    return (1.0 - ((double) wrong / ((Nx + 2) * (Ny + 2) * (Nz + 2))));
}

double Plasma::CheckGPUArraySilent(double *a, double *d_a) {
    static double *t;
    static int f = 1;
    cudaError_t err;


    if (f == 1) {
        t = (double *) malloc(sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2));
        f = 0;
    }
    MemoryCopy(t, d_a, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CheckArraySilent err %d %s \n", err, cudaGetErrorString(err));
        exit(0);
    }

    return CheckArraySilent(a, t, (Nx + 2) * (Ny + 2) * (Nz + 2));
}

int Plasma::CheckValue(double *a, double *dbg_a, int n) {
    Cell c = (*AllCells)[0];

    if (fabs(a[n] - dbg_a[n]) > TOLERANCE) {

        int3 i = c.getCellTripletNumber(n);
#ifdef CHECK_VALUE_DEBUG_PRINTS
        printf("value n %5d i %3d l %3d k %3d %15.5e dbg %1.5e diff %15.5e \n",n,i.x,i.y,i.z,a[n],dbg_a[n],fabs(a[n] - dbg_a[n]));
#endif

        return 0;
    }

    return 1;
}

void Plasma::read3DarrayLog(char *name, double *d, int offset, int col) {
    char str[1000];
    Cell c = (*AllCells)[0];
    FILE *f;

    if ((f = fopen(name, "rt")) == NULL) return;

    while (fgets(str, 1000, f) != NULL) {

        int i = atoi(str + offset) - 1;
        int l = atoi(str + offset + 5) - 1;
        int k = atoi(str + offset + 10) - 1;
        double t = atof(str + offset + 15 + col * 25);
        int n = c.getGlobalCellNumber(i, l, k);
        d[n] = t;
#ifdef READ_ARRAY_LOG_PRINTS
        printf("%d %d %5d %5d %15.5e \n",i,l,k,n,t);
#endif
    }

    fclose(f);
}

void Plasma::read3Darray(char *name, double *d) {
    char str[100];
    Cell c = (*AllCells)[0];
    FILE *f;

    if ((f = fopen(name, "rt")) == NULL) return;

    while (fgets(str, 100, f) != NULL) {
        int i = atoi(str);
        int l = atoi(str + 10);
        int k = atoi(str + 20);
        double t = atof(str + 30);
        int i1, l1, k1, n = c.getFortranCellNumber(i, l, k);
        c.getFortranCellTriplet(n, &i1, &l1, &k1);
        d[n] = t;
    }

    fclose(f);
}

void Plasma::read3Darray(string name, double *d) {
    char str[100];

    strcpy(str, name.c_str());
    read3Darray(str, d);
}

int Plasma::PeriodicCurrentBoundaries(double *E, int dirE, int dir, int start1, int end1, int start2, int end2) {
    Cell c = (*AllCells)[0];

    int N = getBoundaryLimit(dir);

    for (int k = start2; k <= end2; k++) {
        for (int i = start1; i <= end1; i++) {
            int n1 = c.getGlobalBoundaryCellNumber(i, k, dir, 1);
            int n_Nm1 = c.getGlobalBoundaryCellNumber(i, k, dir, N - 1);
#ifdef DEBUG_PLASMA
            int3 n1_3 = c.getCellTripletNumber(n1);
            int3 n_Nm1_3 = c.getCellTripletNumber(n_Nm1);
#endif
            if (dir != dirE) {
                E[n1] += E[n_Nm1];


            }
            if (dir != 1 || dirE != 1) {
                E[n_Nm1] = E[n1];
            }
            int n_Nm2 = c.getGlobalBoundaryCellNumber(i, k, dir, N - 2);
            int n0 = c.getGlobalBoundaryCellNumber(i, k, dir, 0);
#ifdef DEBUG_PLASMA
            int3 n0_3 = c.getCellTripletNumber(n0);
            int3 n_Nm2_3 = c.getCellTripletNumber(n_Nm2);
#endif

            E[n0] += E[n_Nm2];

            E[n_Nm2] = E[n0];
            //   E[n0] = E[n_Nm2];
            //   E[n_Nm1] = E[n1];
            // }
        }
    }

    return 0;
}

void Plasma::ClearAllParticles(void) {
    for (int n = 0; n < (*AllCells).size(); n++) {
        Cell c = (*AllCells)[n];
        c.ClearParticles();
    }
}

int Plasma::initControlPointFile() {
    f_prec_report = fopen("control_points.dat", "wt");
    fclose(f_prec_report);

    return 0;
}

int Plasma::copyCellsWithParticlesToGPU() {
    Cell c000 = (*AllCells)[0];
    magf = 1;
    cudaError_t err;

    int size = (Nx + 2) * (Ny + 2) * (Nz + 2);

    cp = (GPUCell **) malloc(size * sizeof(GPUCell *));
    if ((err = cudaGetLastError()) != cudaSuccess) {
        printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err));
    }

    for (int i = 0; i < size; i++) {
        GPUCell c, *d_c;
        d_c = c.allocateCopyCellFromDevice();
        if ((err = cudaGetLastError()) != cudaSuccess) {
            printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err));
        }

        cp[i] = d_c;
    }
    if ((err = cudaGetLastError()) != cudaSuccess) {
        printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err));
    }
    return 0;
}

void Plasma::ListAllParticles(int nt, std::string where) {
#ifndef LIST_ALL_PARTICLES
    return;
#endif
}

double Plasma::TryCheckCurrent(int nt, double *npJx) {
    return 1.0;    //t_hx;
}

double Plasma::checkNonPeriodicCurrents(int nt) {
    printf("CHECKING Non-periodic currents !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");

    TryCheckCurrent(nt, npJx);

    return 1.0;    //(t_hx+t_hy+t_hz)/3.0;
}

double Plasma::checkPeriodicCurrents(int nt) {

    printf("CHECKING periodic currents !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");

    TryCheckCurrent(nt, Jx);

    return 1.0;    //(t_hx+t_hy+t_hz)/3.0;
}

int Plasma::SetCurrentArraysToZero() {
    cudaError_t err = cudaGetLastError();
    err = cudaGetLastError();
    printf("%s: %d [%d,%d,%d]\n", __FILE__, __LINE__, Nx, Ny, Nz);
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }
    memset(Jx, 0, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2));
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }
    memset(Jy, 0, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2));
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }
    memset(Jz, 0, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2));
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }
    cudaMemset(d_Jx, 0, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2));
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }
    cudaMemset(d_Jy, 0, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2));
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }
    cudaMemset(d_Jz, 0, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2));
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }
    return 0;
}

int Plasma::SetCurrentsInCellsToZero(int nt) {
    dim3 dimGrid(Nx + 2, Ny + 2, Nz + 2), dimBlockExt(CellExtent, CellExtent, CellExtent);
    char name[100];
    sprintf(name, "before_set_to_zero_%03d.dat", nt);

    write3D_GPUArray(name, d_Jx);

    void *args[] = {(void *) &d_CellArray, 0};
    cudaError_t cudaStatus = cudaLaunchKernel(
            (const void *) GPU_SetAllCurrentsToZero, // pointer to kernel func.
            dimGrid,                       // grid
            dimBlockExt,                   // block
            args,                          // arguments
            16000,
            0
    );

    return 0;
}

int Plasma::StepAllCells_fore_diagnostic(int nt) {
    char name[100];

    memory_monitor("CellOrder_StepAllCells3", nt);

    sprintf(name, "before_step_%03d.dat", nt);
    write3D_GPUArray(name, d_Jx);
    ListAllParticles(nt, "bStepAllCells");

    return 0;
}

int Plasma::StepAllCells(int nt, double mass, double q_mass) {
    dim3 dimGrid(Nx + 2, Ny + 2, Nz + 2), dimBlock(512, 1, 1);
    cudaDeviceSynchronize();
    puts("begin step");

    void *args[] = {(void *) &d_CellArray, 0};

    cudaError_t cudaStatus = cudaLaunchKernel(
            (const void *) GPU_StepAllCells, // pointer to kernel func.
            dimGrid,                       // grid
            dimBlock,                      // block
            args,                          // arguments
            16000,
            0
    );


    std::cout << "GPU_StepAllCells returns " << cudaStatus << std::endl;
    dim3 dimBlock1(1, 1, 1);
    void *args1[] = {(void *) &d_CellArray, &nt, 0};
    cudaStatus = cudaFuncSetCacheConfig((const void *) GPU_CurrentsAllCells, cudaFuncCachePreferShared);
    std::cout << "cudaFuncSetCacheConfig returns " << cudaStatus << " " << cudaGetErrorString(cudaStatus) << std::endl;
    cudaStatus = cudaLaunchKernel(
            (const void *) GPU_CurrentsAllCells, // pointer to kernel func.
            dimGrid,                             // grid
            dimBlock,                            // block
            args1,                               // arguments
            4000,
            0
    );
    std::cout << "GPU_CurrentsAllCells returns " << cudaStatus << " " << cudaGetErrorString(cudaStatus) << std::endl;
    puts("end step");
    cudaDeviceSynchronize();

    puts("end step-12");

    return 0;
}

int Plasma::StepAllCells_post_diagnostic(int nt) {
    memory_monitor("CellOrder_StepAllCells4", nt);

    ListAllParticles(nt, "aStepAllCells");
    cudaError_t err2 = cudaGetLastError();
    char err_s[200];
    strcpy(err_s, cudaGetErrorString(err2));

    return (int) err2;
}

int Plasma::WriteCurrentsFromCellsToArrays(int nt) {
    char name[100];
    dim3 dimGrid(Nx + 2, Ny + 2, Nz + 2);

    sprintf(name, "before_write_currents_%03d.dat", nt);
    write3D_GPUArray(name, d_Jx);

    dim3 dimExt(CellExtent, CellExtent, CellExtent);

    int zero = 0;
    void *args[] = {(void *) &d_CellArray,
                    (void *) &zero,
                    (void *) &d_Jx,
                    (void *) &d_Jy,
                    (void *) &d_Jz,
                    (void *) &d_Rho,
                    0};
    cudaError_t cudaStatus = cudaLaunchKernel(
            (const void *) GPU_WriteAllCurrents, // pointer to kernel func.
            dimGrid,                             // grid
            dimExt,                              // block
            args,                                // arguments
            16000,
            0
    );

    memory_monitor("CellOrder_StepAllCells5", nt);

    sprintf(name, "after_write_currents_%03d.dat", nt);
    write3D_GPUArray(name, d_Jx);

    memory_monitor("CellOrder_StepAllCells6", nt);

    return 0;
}

int Plasma::MakeParticleList(int nt, int *stage, int *stage1, int **d_stage, int **d_stage1) {
    dim3 dimGrid(Nx + 2, Ny + 2, Nz + 2), dimGridOne(1, 1, 1), dimBlock(512, 1, 1),
            dimBlockOne(1, 1, 1), dimBlockGrow(1, 1, 1), dimBlockExt(CellExtent, CellExtent, CellExtent);
    dim3 dimGridBulk(Nx, Ny, Nz);
    cudaError_t before_MakeDepartureLists, after_MakeDepartureLists;

    before_MakeDepartureLists = cudaGetLastError();
    printf("before_MakeDepartureLists %d %s blockdim %d %d %d\n", before_MakeDepartureLists,
           cudaGetErrorString(before_MakeDepartureLists), dimGrid.x, dimGrid.y, dimGrid.z);

    cudaMalloc((void **) d_stage, sizeof(int) * (Nx + 2) * (Ny + 2) * (Nz + 2));

    cudaMalloc((void **) d_stage1, sizeof(int) * (Nx + 2) * (Ny + 2) * (Nz + 2));

    void *args[] = {
            (void *) &d_CellArray,
            (void *) &nt,
            (void *) d_stage,
            0};
    cudaError_t cudaStatus = cudaLaunchKernel(
            (const void *) GPU_MakeDepartureLists, // pointer to kernel func.
            dimGrid,                               // grid
            dimBlockOne,                           // block
            args,                                  // arguments
            16000,
            0
    );

    after_MakeDepartureLists = cudaGetLastError();
    if (after_MakeDepartureLists != cudaSuccess) {
        printf("after_MakeDepartureLists %d %s\n", after_MakeDepartureLists,
               cudaGetErrorString(after_MakeDepartureLists));
    }

    cudaDeviceSynchronize();

    int err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("MakeParticleList sync error %d %s\n", err, getErrorString(err));
    }
    err = MemoryCopy(stage, *d_stage, sizeof(int) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);

    if (err != cudaSuccess) {
        printf("MakeParticleList error %d %s\n", err, getErrorString(err));
        exit(0);
    }

    return (int) err;
}

int Plasma::inter_stage_diagnostic(int *stage, int nt) {
    if (stage[0] == TOO_MANY_PARTICLES) {
        printf("too many particles flying to (%d,%d,%d) from (%d,%d,%d) \n", stage[1], stage[2], stage[3], stage[4],
               stage[5], stage[6]);
        exit(0);
    }

    ListAllParticles(nt, "aMakeDepartureLists");
#ifdef BALANCING_PRINTS
    before_ArrangeFlights = cudaGetLastError();
    printf("before_ArrangeFlights %d %s\n",before_ArrangeFlights,cudaGetErrorString(before_ArrangeFlights));
#endif

    return 0;
}

int Plasma::reallyPassParticlesToAnotherCells(int nt, int *stage1, int *d_stage1) {
    int err;
    dim3 dimGridBulk(Nx, Ny, Nz), dimBlockOne(1, 1, 1);
    cudaMemset(d_stage1, 0, sizeof(int) * (Nx + 2) * (Ny + 2) * (Nz + 2));

    void *args[] = {
            (void *) &d_CellArray,
            (void *) &nt,
            (void *) &d_stage1,
            0};

    cudaError_t cudaStatus = cudaLaunchKernel(
            (const void *) GPU_ArrangeFlights, // pointer to kernel func.
            dimGridBulk,                       // grid
            dimBlockOne,                       // block
            args,                              // arguments
            16000,
            0
    );


#ifdef BALANCING_PRINTS
    CUDA_Errot_t after_ArrangeFlights = cudaGetLastError();
    printf("after_ArrangeFlights %d %s\n",after_ArrangeFlights,cudaGetErrorString(after_ArrangeFlights));
    cudaDeviceSynchronize();
#endif

    err = MemoryCopy(stage1, d_stage1, sizeof(int) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    if (err != cudaSuccess) {
        puts("copy error");
        exit(0);
    }
    ListAllParticles(nt, "aArrangeFlights");

    memory_monitor("CellOrder_StepAllCells7", nt);
    return (int) err;
}

int Plasma::reorder_particles(int nt) {
    int stage[4000], stage1[4000], *d_stage, *d_stage1, err;

    MakeParticleList(nt, stage, stage1, &d_stage, &d_stage1);

    inter_stage_diagnostic(stage, nt);

    err = reallyPassParticlesToAnotherCells(nt, stage1, d_stage1);

    return (int) err;
}

int Plasma::Push(int nt, double mass, double q_mass) {
    StepAllCells_fore_diagnostic(nt);

    StepAllCells(nt, mass, q_mass);
    puts("after StepAllCell");

    return StepAllCells_post_diagnostic(nt);
}

int Plasma::SetCurrentsToZero(int nt) {
    cudaError_t err = cudaGetLastError();
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }
    SetCurrentArraysToZero();

    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }

    return SetCurrentsInCellsToZero(nt);
}

void Plasma::CellOrder_StepAllCells(int nt, double mass, double q_mass, int first) {
    cudaError_t err = cudaGetLastError();
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }
    SetCurrentsToZero(nt);

    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }

    Push(nt, mass, q_mass);
    puts("Push");
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }

    WriteCurrentsFromCellsToArrays(nt);
    puts("writeCut2arr");
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }

    reorder_particles(nt);
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); }
}

double Plasma::checkControlPointParticlesOneSort(int check_point_num, FILE *f, GPUCell **copy_cells, int nt, int sort) {

    double t = 0.0;
    int size = 1;
#ifdef CPU_DEBUG_RUN
    double q_m, m;

    memory_monitor("checkControlPointParticlesOneSort", nt);

    Cell c0 = (*AllCells)[0];

    total_particles = readBinaryParticleArraysOneSort(f, &dbg_x, &dbg_y, &dbg_z,
                                                      &dbg_px, &dbg_py, &dbg_pz, &q_m, &m, nt, sort);
    memory_monitor("checkControlPointParticlesOneSort2", nt);

    size = (*AllCells).size();

    for (int i = 0; i < size; i++) {
        GPUCell c = *(copy_cells[i]);

#ifdef checkControlPointParticles_PRINT
        printf("cell %d particles %20d \n",i,c.number_of_particles);
#endif

        t += c.checkCellParticles(check_point_num, dbg_x, dbg_y, dbg_z, dbg_px, dbg_py, dbg_pz, q_m, m);

    }
    memory_monitor("checkControlPointParticlesOneSort3", nt);

    free(dbg_x);
    free(dbg_y);
    free(dbg_z);

    free(dbg_px);
    free(dbg_py);
    free(dbg_pz);
    memory_monitor("checkControlPointParticlesOneSort4", nt);
#endif
    return t / size;
}

double Plasma::checkControlPointParticles(int check_point_num, FILE *f, char *fname, int nt) {
    double te = 0.0, ti = 0.0, tb = 0.0;
    struct sysinfo info;
#ifdef CPU_DEBUG_RUN

    int size = (*AllCells).size();

    char where[100];
    sprintf(where, "checkpoint%03d", check_point_num);
    copyCells(where, nt);

#ifdef FREE_RAM_MONITOR
    sysinfo(&info);
#ifdef checkControlPointParticles_PRINTS
    printf("checkControlPointParticles %u \n",info.freeram/1024/1024);
#endif
#endif

    GPUCell c = *(cp[141]);
#ifdef checkControlPointParticles_PRINTS
    printf("checkControlPointParticlesOneSort cell 141 particles %20d \n",c.number_of_particles);
#endif

#ifdef FREE_RAM_MONITOR
    sysinfo(&info);
#ifdef checkControlPointParticles_PRINTS
    printf("checkControlPointParticles0.9 %u \n",info.freeram/1024/1024);
#endif
#endif

    ti = checkControlPointParticlesOneSort(check_point_num, f, cp, nt, ION);
#ifdef FREE_RAM_MONITOR
    sysinfo(&info);
#ifdef checkControlPointParticles_PRINTS
    printf("checkControlPointParticles1 %u \n",info.freeram/1024/1024);
#endif
#endif

    te = checkControlPointParticlesOneSort(check_point_num, f, cp, nt, PLASMA_ELECTRON);

#ifdef FREE_RAM_MONITOR
    sysinfo(&info);
#ifdef checkControlPointParticles_PRINTS
    printf("checkControlPointParticles1.5 %u \n",info.freeram/1024/1024);
#endif
#endif

    tb = checkControlPointParticlesOneSort(check_point_num, f, cp, nt, BEAM_ELECTRON);

#ifdef FREE_RAM_MONITOR
    sysinfo(&info);
#ifdef checkControlPointParticles_PRINTS
    printf("checkControlPointParticles2 %u \n",info.freeram/1024/1024);
#endif
#endif

#endif

    memory_monitor("after_free", nt);
    return (te + ti + tb) / 3.0;
}

int Plasma::readControlFile(int nt) {

#ifndef ATTRIBUTES_CHECK
    return 0;
#else
    FILE *f;
    char fname[100];
    static int first = 1;
    int size;

    sprintf(fname,"ctrl%05d",nt);

    if((f = fopen(fname,"rb")) == NULL) {
        puts("no ini-file");
        exit(0);
    }

    fread(&size,sizeof(int),1,f);
    fread(&ami,sizeof(double),1,f);
    fread(&amf,sizeof(double),1,f);
    fread(&amb,sizeof(double),1,f);
    fread(&size,sizeof(int),1,f);

    fread(&size,sizeof(int),1,f);

    if(first == 1) {
        first = 0;
        ctrlParticles = (double *)malloc(size);
#ifdef ATTRIBUTES_CHECK
        memset(ctrlParticles,0,size);
        cudaMalloc((void **)&d_ctrlParticles,size);
        cudaMemset(d_ctrlParticles,0,size);
        size_ctrlParticles = size;
#endif
    }
    fread(ctrlParticles,1,size,f);


    jmp = size/sizeof(double)/PARTICLE_ATTRIBUTES/3;

    return 0;
#endif
}

int Plasma::memory_monitor(std::string legend, int nt) {
    static int first = 1;
    static FILE *f;

#ifndef FREE_RAM_MONITOR
    return 1;
#endif

    if (first == 1) {
        first = 0;
        f = fopen("memory_monitor.log", "wt");
    }

    size_t m_free, m_total;
    struct sysinfo info;


    cudaError_t err = cudaMemGetInfo(&m_free, &m_total);

    sysinfo(&info);                                                                //  1   2              3                 4                5
    fprintf(f, "step %10d %50s GPU memory total %10d free %10d free CPU memory %10u \n", nt, legend.c_str(),
            ((int) m_total) / 1024 / 1024, ((int) m_free) / 1024 / 1024, ((int) info.freeram) / 1024 / 1024);

    return 0;
}

int Plasma::memory_status_print(int nt) {
    size_t m_free, m_total;
    struct sysinfo info;

    cudaMemGetInfo(&m_free, &m_total);
    sysinfo(&info);

#ifdef MEMORY_PRINTS
    printf("before Step  %10d CPU memory free %10u GPU memory total %10d free %10d\n", nt,info.freeram/1024/1024,m_total/1024/1024,m_free/1024/1024);
#endif

    return 0;
}

/**
 * @param step {int} - save data to file
 */
void Plasma::writeDataToFile(int step) {
    std::string step_str;
    std::stringstream ss;
    ss << step;
    step_str = ss.str();
    string filename = dataFileStartPattern + step_str + dataFileEndPattern;

    NcFile dataFile(filename, NcFile::replace);
    dataFile.close();

    // copy dimensions
    NetCDFManipulator::plsm_add_dimension(filename.c_str(), "x", NX);
    NetCDFManipulator::plsm_add_dimension(filename.c_str(), "y", NY);
    NetCDFManipulator::plsm_add_dimension(filename.c_str(), "z", NZ);

    double *t = (double *) malloc(sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2));
    int err;

    err = MemoryCopy(t, d_Ex, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    writeOne3DArray(filename.c_str(), t, ELECTRIC_FIELD_LABEL + X_LABEL, UNITS_ELECTRIC_FIELD,
                    DESC_ELECTRIC_FIELD + ELECTRIC_FIELD_LABEL + X_LABEL);
    err = MemoryCopy(t, d_Ey, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    writeOne3DArray(filename.c_str(), t, ELECTRIC_FIELD_LABEL + Y_LABEL, UNITS_ELECTRIC_FIELD,
                    DESC_ELECTRIC_FIELD + ELECTRIC_FIELD_LABEL + Y_LABEL);
    err = MemoryCopy(t, d_Ez, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    writeOne3DArray(filename.c_str(), t, ELECTRIC_FIELD_LABEL + Z_LABEL, UNITS_ELECTRIC_FIELD,
                    DESC_ELECTRIC_FIELD + ELECTRIC_FIELD_LABEL + Z_LABEL);

    err = MemoryCopy(t, d_Hx, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    writeOne3DArray(filename.c_str(), t, MAGNETIC_FIELD_LABEL + X_LABEL, UNITS_MAGNETIC_FIELD,
                    DESC_MAGNETIC_FIELD + MAGNETIC_FIELD_LABEL + X_LABEL);
    err = MemoryCopy(t, d_Hy, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    writeOne3DArray(filename.c_str(), t, MAGNETIC_FIELD_LABEL + Y_LABEL, UNITS_MAGNETIC_FIELD,
                    DESC_MAGNETIC_FIELD + MAGNETIC_FIELD_LABEL + Y_LABEL);
    err = MemoryCopy(t, d_Hz, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    writeOne3DArray(filename.c_str(), t, MAGNETIC_FIELD_LABEL + Z_LABEL, UNITS_MAGNETIC_FIELD,
                    DESC_MAGNETIC_FIELD + MAGNETIC_FIELD_LABEL + Z_LABEL);

    err = MemoryCopy(t, d_Jx, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    writeOne3DArray(filename.c_str(), t, CURRENT_FIELD_LABEL + X_LABEL, UNITS_NO,
                    CURRENT + CURRENT_FIELD_LABEL + X_LABEL);
    err = MemoryCopy(t, d_Jy, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    writeOne3DArray(filename.c_str(), t, CURRENT_FIELD_LABEL + Y_LABEL, UNITS_NO,
                    CURRENT + CURRENT_FIELD_LABEL + Y_LABEL);
    err = MemoryCopy(t, d_Jz, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    writeOne3DArray(filename.c_str(), t, CURRENT_FIELD_LABEL + Z_LABEL, UNITS_NO,
                    CURRENT + CURRENT_FIELD_LABEL + Z_LABEL);

    err = MemoryCopy(t, d_Qx, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    writeOne3DArray(filename.c_str(), t, MAGNETIC_HALF_STEP_FIELD_LABEL + X_LABEL, UNITS_NO,
                    DESC_HALFSTEP + MAGNETIC_HALF_STEP_FIELD_LABEL + X_LABEL);
    err = MemoryCopy(t, d_Qy, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    writeOne3DArray(filename.c_str(), t, MAGNETIC_HALF_STEP_FIELD_LABEL + Y_LABEL, UNITS_NO,
                    DESC_HALFSTEP + MAGNETIC_HALF_STEP_FIELD_LABEL + Y_LABEL);
    err = MemoryCopy(t, d_Qz, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    writeOne3DArray(filename.c_str(), t, MAGNETIC_HALF_STEP_FIELD_LABEL + Z_LABEL, UNITS_NO,
                    DESC_HALFSTEP + MAGNETIC_HALF_STEP_FIELD_LABEL + Z_LABEL);

    free(t);

    if (err != cudaSuccess) {
        printf("bCheckArray err %d %s \n", err, getErrorString(err));
        exit(0);
    }
}

/**
 * @param step {int} - step number
 */
void Plasma::Step(int step) {
    ComputeField_FirstHalfStep(step);

    PushParticles(step);

    puts("push ended");

    ComputeField_SecondHalfStep(step);
    puts("field computed-2");

    sumMPI((Nx + 2) * (Ny + 2) * (Nz + 2), d_Jx, d_Jy, d_Jz);

    Diagnose(step);
}

/**
 * @param startStep {int} - Value of step to start from
 * @param totalSteps {int} - Number of total steps
 * @param startSaveStep {int} - Start save data on 'startSaveStep' step
 * @param saveStep {int} - Save data every 'saveStep' step
 */
int Plasma::Compute(int startStep, int totalSteps, int startSaveStep, int saveStep) {
    printf("----------------------------------------------------------- \n");
    size_t m_free, m_total;

    cudaMemGetInfo(&m_free, &m_total);

    if (startStep <= 0 || totalSteps <= 0) {
        cout << "Invalid computation parameters values!" << endl;
        return 1;
    }

    for (int step = startStep; step <= totalSteps; step++) {
        memory_status_print(step);

        Step(step);

        // save file
        if (startSaveStep >= 0 && saveStep >= 0) {
            writeDataToFile(step);
        }

        memory_status_print(step);

        printf("step %d ===================\n", step);
    }
    printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n");

    return 0;
}

/**
 * @param startStep {int} - Value of step to start from
 * @param totalSteps {int} - Number of total steps
 */
int Plasma::Compute(int startStep, int totalSteps) {
    return Compute(startStep, totalSteps, 0, 0);
}
