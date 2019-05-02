//
// Created by egor on 19.02.19.
//

#include "../../include/Plasma.h"

Plasma::Plasma(PlasmaConfig *p) {
    this->pd = p;
    this->temp = new double[(pd->nx + 2) * (pd->ny + 2) * (pd->nz + 2)];

    dataFileStartPattern = "data0";
    dataFileEndPattern = ".nc";
}

Plasma::~Plasma() {
    delete[] temp;
    delete[] pd->Ex;
    delete[] pd->Ey;
    delete[] pd->Ez;
    delete[] pd->Hx;
    delete[] pd->Hy;
    delete[] pd->Hz;
    delete[] pd->Jx;
    delete[] pd->Jy;
    delete[] pd->Jz;
    delete[] pd->Rho;

    delete[] pd->npJx;
    delete[] pd->npJy;
    delete[] pd->npJz;

    delete[] pd->Qx;
    delete[] pd->Qy;
    delete[] pd->Qz;

#ifdef DEBUG
    delete[] pd->dbgEx;
    delete[] pd->dbgEy;
    delete[] pd->dbgEz;

    delete[] pd->dbgHx;
    delete[] pd->dbgHy;
    delete[] pd->dbgHz;

    delete[] pd->dbgJx;
    delete[] pd->dbgJy;
    delete[] pd->dbgJz;

    delete[] pd->dbg_Qx;
    delete[] pd->dbg_Qy;
    delete[] pd->dbg_Qz;
#endif
}

void Plasma::copyCells(int nt) {
    static int first = 1;
    size_t m_free, m_total;
    int size = (int)(*pd->AllCells).size();
    struct sysinfo info;

    if (first == 1) {
        pd->cp = (GPUCell **) malloc(size * sizeof(GPUCell *));
    }

    unsigned long m1, m2, delta, accum = 0;
    memory_monitor("beforeCopyCells", nt);

    for (int i = 0; i < size; i++) {
        int err = cudaMemGetInfo(&m_free, &m_total);
        CHECK_ERROR("MEM GET INFO", err);

        sysinfo(&info);
        m1 = info.freeram;
        GPUCell c, *d_c, *z0;
        z0 = pd->h_CellArray[i];
        if (first == 1) {
            d_c = c.allocateCopyCellFromDevice();
            pd->cp[i] = d_c;
        } else {
            d_c = pd->cp[i];
        }
        c.copyCellFromDevice(z0, d_c);
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
    char fname[1000];
    double res;
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;

#ifndef CHECK_ARRAY_OUTPUT
    return 0.0;
#endif

    sprintf(fname, "diff_%s_at_%s_nt%08d.dat", name.c_str(), where.c_str(), nt);

    int err;
    err = MemoryCopy(this->temp, d_a, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    CHECK_ERROR("MEM COPY", err);

    int size = (Nx + 2) * (Ny + 2) * (Nz + 2);
    res = CheckArraySilent(a, this->temp, size);

    return res;
}

void Plasma::emeGPUIterate(int3 s, int3 f, double *E, double *H1, double *H2, double *J, double c1, double c2, double tau, int3 d1, int3 d2) {
    dim3 dimGrid(f.x - s.x + 1, 1, 1), dimBlock(1, f.y - s.y + 1, f.z - s.z + 1);

    void *args[] = {(void *) &pd->d_CellArray,
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

    int cudaStatus = cudaLaunchKernel(
            (const void *) GPU_eme,        // pointer to kernel func.
            dimGrid,                       // grid
            dimBlock,                      // block
            args,                          // arguments
            0,
            0
    );

    CHECK_ERROR("Launch kernel", cudaStatus);

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

int Plasma::ElectricFieldTrace(double *E, double *H1, double *H2, double *J, int dir, double c1, double c2, double tau) {
    int3 start, d1, d2, finish = make_int3(pd->nx, pd->ny, pd->nz);

    GetElectricFieldStartsDirs(&start, &d1, &d2, dir);

    emeGPUIterate(start, finish, E, H1, H2, J, c1, c2, tau, d1, d2);

    return 0;
}

int Plasma::checkFields_beforeMagneticStageOne(int nt) {
    memory_monitor("beforeComputeField_FirstHalfStep", nt);

    return 0;
}

int Plasma::checkFields_afterMagneticStageOne(int nt) {
    checkControlPoint(50, nt);
    memory_monitor("afterComputeField_FirstHalfStep", nt);

    return 0;
}

void Plasma::ComputeField_FirstHalfStep(int nt) {
    checkFields_beforeMagneticStageOne(nt);

    MagneticStageOne(pd->d_Qx, pd->d_Qy, pd->d_Qz, pd->d_Hx, pd->d_Hy, pd->d_Hz, pd->d_Ex, pd->d_Ey, pd->d_Ez);

    checkFields_afterMagneticStageOne(nt);

    AssignCellsToArraysGPU();
}

void Plasma::ComputeField_SecondHalfStep(int nt) {
    SetPeriodicCurrents(nt);
    MagneticFieldStageTwo(pd->d_Hx, pd->d_Hy, pd->d_Hz, nt, pd->d_Qx, pd->d_Qy, pd->d_Qz);
    ElectricFieldEvaluate(pd->d_Ex, pd->d_Ey, pd->d_Ez, nt, pd->d_Hx, pd->d_Hy, pd->d_Hz, pd->d_Jx, pd->d_Jy, pd->d_Jz);
}

void Plasma::ElectricFieldComponentEvaluateTrace(double *E, double *H1, double *H2, double *J, int dir, double c1, double c2, double tau) {
    ElectricFieldTrace(E, H1, H2, J, dir, c1, c2, tau);
}

void Plasma::ElectricFieldComponentEvaluatePeriodic(double *E, int dir, int dir_1, int start1_1, int end1_1, int start2_1, int end2_1, int N_1, int dir_2, int start1_2, int end1_2, int start2_2, int end2_2, int N_2) {
    if (dir != 0) {
        PeriodicBoundaries(E, dir_1, start1_1, end1_1, start2_1, end2_1, N_1);
    }

    if (dir != 2) {
        PeriodicBoundaries(E, dir_2, start1_2, end1_2, start2_2, end2_2, N_2);
    }
}

void Plasma::ElectricFieldEvaluate(double *locEx, double *locEy, double *locEz, int nt, double *locHx, double *locHy, double *locHz, double *loc_npJx, double *loc_npJy, double *loc_npJz) {
    double3 c1 = getMagneticFieldTimeMeshFactors();
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;

    ElectricFieldComponentEvaluateTrace(locEx, locHz, locHy, loc_npJx, 0, c1.y, c1.z, pd->tau);
    ElectricFieldComponentEvaluateTrace(locEy, locHx, locHz, loc_npJy, 1, c1.z, c1.x, pd->tau);
    ElectricFieldComponentEvaluateTrace(locEz, locHy, locHx, loc_npJz, 2, c1.x, c1.y, pd->tau);

    checkControlPoint(550, nt);

    ElectricFieldComponentEvaluatePeriodic(locEx, 0, 1, 0, Nx, 1, Nz, Ny, 2, 0, Nx, 0, Ny + 1, Nz);
    ElectricFieldComponentEvaluatePeriodic(locEy, 1, 0, 0, Ny, 1, Nz, Nx, 2, 0, Nx + 1, 0, Ny, Nz);
    ElectricFieldComponentEvaluatePeriodic(locEz, 2, 0, 1, Ny, 0, Nz, Nx, 1, 0, Nx + 1, 0, Nz, Ny);

    checkControlPoint(600, nt);

    memory_monitor("after_ComputeField_SecondHalfStep", nt);
}

double3 Plasma::getMagneticFieldTimeMeshFactors() {
    Cell c = (*pd->AllCells)[0];
    double hx = c.get_hx(), hy = c.get_hy(), hz = c.get_hz();
    double3 d;
    d.x = pd->tau / (hx);
    d.y = pd->tau / (hy);
    d.z = pd->tau / hz;

    return d;
}

void Plasma::MagneticStageOne(double *Qx, double *Qy, double *Qz, double *Hx, double *Hy, double *Hz, double *Ex, double *Ey, double *Ez) {
    double3 c1 = getMagneticFieldTimeMeshFactors();
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;

    MagneticFieldTrace(Qx, Hx, Ey, Ez, Nx + 1, Ny, Nz, c1.z, c1.y, 0);
    MagneticFieldTrace(Qy, Hy, Ez, Ex, Nx, Ny + 1, Nz, c1.x, c1.z, 1);
    MagneticFieldTrace(Qz, Hz, Ex, Ey, Nx, Ny, Nz + 1, c1.y, c1.x, 2);

}

void Plasma::MagneticFieldStageTwo(double *Hx, double *Hy, double *Hz, int nt, double *Qx, double *Qy, double *Qz) {
    Cell c = (*pd->AllCells)[0];
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;

    SimpleMagneticFieldTrace(c, Qx, Hx, Nx + 1, Ny, Nz);
    SimpleMagneticFieldTrace(c, Qy, Hy, Nx, Ny + 1, Nz);
    SimpleMagneticFieldTrace(c, Qz, Hz, Nx, Ny, Nz + 1);

    checkControlPoint(500, nt);
}

int Plasma::PushParticles(int nt) {
    memory_monitor("before_CellOrder_StepAllCells", nt);

    CellOrder_StepAllCells(nt);
    std::cout << "cell_order" << std::endl;

    memory_monitor("after_CellOrder_StepAllCells", nt);

    checkControlPoint(270, nt);

    return 0;
}

double Plasma::getElectricEnergy() {
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;
    dim3 dimGrid((unsigned int)(Nx + 2), (unsigned int)(Ny + 2), (unsigned int)(Nz + 2)), dimBlockOne(1, 1, 1);
    static int first = 1;
    static double *d_ee;
    double ee;

    if (first == 1) {
        cudaMalloc((void **) &d_ee, sizeof(double));
        first = 0;
    }

    cudaMemset(d_ee, 0, sizeof(double));

    void *args[] = {(void *) &pd->d_CellArray,
                    (void *) &d_ee,
                    (void *) &pd->d_Ex,
                    (void *) &pd->d_Ey,
                    (void *) &pd->d_Ez,
                    0};

    int cudaStatus = cudaLaunchKernel(
            (const void *) GPU_getCellEnergy, // pointer to kernel func.
            dimGrid,                          // grid
            dimBlockOne,                      // block
            args,                             // arguments
            0,
            0
    );

    CHECK_ERROR("Launch kernel", cudaStatus);

    cudaStatus = MemoryCopy(&ee, d_ee, sizeof(double), DEVICE_TO_HOST);

    CHECK_ERROR("MEM COPY", cudaStatus);

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

int Plasma::getMagneticFieldTraceShifts(int dir, int3 &d1, int3 &d2) {
    d1.x = (dir == 0) * 0 + (dir == 1) * 1 + (dir == 2) * 0;
    d1.y = (dir == 0) * 0 + (dir == 1) * 0 + (dir == 2) * 1;
    d1.z = (dir == 0) * 1 + (dir == 1) * 0 + (dir == 2) * 0;

    d2.x = (dir == 0) * 0 + (dir == 1) * 0 + (dir == 2) * 1;
    d2.y = (dir == 0) * 1 + (dir == 1) * 0 + (dir == 2) * 0;
    d2.z = (dir == 0) * 0 + (dir == 1) * 1 + (dir == 2) * 0;

    return 0;
}

int Plasma::MagneticFieldTrace(double *Q, double *H, double *E1, double *E2, int i_end, int l_end, int k_end, double c1, double c2, int dir) {
    int3 d1, d2;

    getMagneticFieldTraceShifts(dir, d1, d2);

    dim3 dimGrid(i_end + 1, l_end + 1, k_end + 1), dimBlock(1, 1, 1);

    void *args[] = {(void *) &pd->d_CellArray,
                    (void *) &Q,
                    (void *) &H,
                    (void *) &E1,
                    (void *) &E2,
                    (void *) &c1,
                    (void *) &c2,
                    (void *) &d1,
                    (void *) &d2,
                    0};

    int cudaStatus = cudaLaunchKernel(
            (const void *) GPU_emh1, // pointer to kernel func.
            dimGrid,                 // grid
            dimBlock,                // block
            args,                    // arguments
            0,
            0
    );

    CHECK_ERROR("Launch kernel", cudaStatus);

    return 0;
}

int Plasma::SimpleMagneticFieldTrace(Cell &c, double *Q, double *H, int i_end, int l_end, int k_end) {
    dim3 dimGrid(i_end + 1, l_end + 1, k_end + 1), dimBlock(1, 1, 1);

    int i_s = 0;
    int l_s = 0;
    int k_s = 0;

    void *args[] = {(void *) &pd->d_CellArray,
                    (void *) &i_s,
                    (void *) &l_s,
                    (void *) &k_s,
                    (void *) &Q,
                    (void *) &H,
                    0};

    int cudaStatus = cudaLaunchKernel(
            (const void *) GPU_emh2,       // pointer to kernel func.
            dimGrid,                       // grid
            dimBlock,                      // block
            args,                          // arguments
            0,
            0
    );

    CHECK_ERROR("Launch kernel", cudaStatus);

    return 0;
}

int Plasma::PeriodicBoundaries(double *E, int dir, int start1, int end1, int start2, int end2, int N) {
    Cell c = (*pd->AllCells)[0];

    dim3 dimGrid(end1 - start1 + 1, 1, end2 - start2 + 1), dimBlock(1, 1, 1);

    int zero = 0;
    void *args[] = {(void *) &pd->d_CellArray,
                    (void *) &start1,
                    (void *) &start2,
                    (void *) &E,
                    (void *) &dir,
                    (void *) &zero,
                    (void *) &N,
                    0};

    int cudaStatus = cudaLaunchKernel(
            (const void *) GPU_periodic,   // pointer to kernel func.
            dimGrid,                       // grid
            dimBlock,                      // block
            args,                          // arguments
            0,
            0
    );

    CHECK_ERROR("Launch kernel", cudaStatus);

    int one = 1;
    int N1 = N + 1;

    void *args1[] = {(void *) &pd->d_CellArray,
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

    CHECK_ERROR("Launch kernel", cudaStatus);

    return 0;
}

int Plasma::SetPeriodicCurrentComponent(GPUCell **cells, double *J, int dir, unsigned int Nx, unsigned int Ny, unsigned int Nz) {
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

    int cudaStatus = cudaLaunchKernel(
            (const void *) GPU_CurrentPeriodic, // pointer to kernel func.
            dimGridX,                           // grid
            dimBlock,                           // block
            args,                               // arguments
            16000,
            0
    );

    CHECK_ERROR("Launch kernel", cudaStatus);

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

    CHECK_ERROR("Launch kernel", cudaStatus);

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

    CHECK_ERROR("Launch kernel", cudaStatus);

    return 0;
}

void Plasma::SetPeriodicCurrents(int nt) {
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;

    memory_monitor("before275", nt);

    checkControlPoint(275, nt);
    SetPeriodicCurrentComponent(pd->d_CellArray, pd->d_Jx, 0, Nx, Ny, Nz);
    SetPeriodicCurrentComponent(pd->d_CellArray, pd->d_Jy, 1, Nx, Ny, Nz);
    SetPeriodicCurrentComponent(pd->d_CellArray, pd->d_Jz, 2, Nx, Ny, Nz);

    checkControlPoint(400, nt);
}

void Plasma::AssignCellsToArraysGPU() {
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;
    int err;
    size_t sz;
    dim3 dimGrid(Nx, Ny, Nz), dimBlockExt(CellExtent, CellExtent, CellExtent);

    err = cudaDeviceGetLimit(&sz, cudaLimitStackSize);
    CHECK_ERROR("DEVICE LIMIT", err);
    printf("%s:%d - stack limit %d\n", __FILE__, __LINE__, ((int) sz));

    err = cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);
    CHECK_ERROR("DEVICE LIMIT", err);
    printf("%s:%d - set stack limit \n", __FILE__, __LINE__);

    err = cudaDeviceGetLimit(&sz, cudaLimitStackSize);
    CHECK_ERROR("DEVICE LIMIT", err);
    printf("%s:%d - stack limit %d \n", __FILE__, __LINE__, ((int) sz));

    void *args[] = {(void *) &pd->d_CellArray, &pd->d_Ex, &pd->d_Ey, &pd->d_Ez, &pd->d_Hx, &pd->d_Hy, &pd->d_Hz, 0};
    err = cudaLaunchKernel(
            (const void *) GPU_SetFieldsToCells, // pointer to kernel func.
            dimGrid,                             // grid
            dimBlockExt,                         // block
            args,                                // arguments
            16000,
            0
    );
    CHECK_ERROR("GPU_SetFieldsToCells", err);
    err = cudaDeviceSynchronize();
    CHECK_ERROR("cudaDeviceSynchronize", err);
}

void Plasma::readControlPoint(const char * fn, int field_assign,
                              double *ex, double *ey, double *ez,
                              double *hx, double *hy, double *hz,
                              double *jx, double *jy, double *jz,
                              double *qx, double *qy, double *qz
) {
    std::cout << "Control point " << " | " <<  "Values to check with are read from file: " << fn << std::endl;

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

    if (field_assign == 1) pi->AssignArraysToCells();
}

void Plasma::checkControlPoint(int num, int nt) {
#ifdef DEBUG
    double t_ex, t_ey, t_ez, t_hx, t_hy, t_hz, t_jx, t_jy, t_jz;
    double t_qx, t_qy, t_qz;
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;

    if ((nt != pd->lst) || (num != 600)) {
#ifndef CONTROL_POINT_CHECK
        return;
#endif
    }

    if (pd->checkFile == NULL || nt % FORTRAN_NUMBER_OF_SMALL_STEPS != 0) return;

    memory_monitor("checkControlPoint1", nt);

    readControlPoint(pd->checkFile, 0,
                     pd->dbgEx, pd->dbgEy, pd->dbgEz,
                     pd->dbgHx, pd->dbgHy, pd->dbgHz,
                     pd->dbgJx, pd->dbgJy, pd->dbgJz,
                     pd->dbg_Qx, pd->dbg_Qy, pd->dbg_Qz);

    memory_monitor("checkControlPoint2", nt);

    int size = (Nx + 2) * (Ny + 2) * (Nz + 2);

    t_ex = CheckArraySilent(pd->Ex, pd->dbgEx, size);
    t_ey = CheckArraySilent(pd->Ey, pd->dbgEy, size);
    t_ez = CheckArraySilent(pd->Ez, pd->dbgEz, size);
    t_hx = CheckArraySilent(pd->Hx, pd->dbgHx, size);
    t_hy = CheckArraySilent(pd->Hy, pd->dbgHy, size);
    t_hz = CheckArraySilent(pd->Hz, pd->dbgHz, size);
    t_jx = CheckArraySilent(pd->Jx, pd->dbgJx, size);
    t_jy = CheckArraySilent(pd->Jy, pd->dbgJy, size);
    t_jz = CheckArraySilent(pd->Jz, pd->dbgJz, size);

    memory_monitor("checkControlPoint3", nt);

    t_ex = CheckGPUArraySilent(pd->dbgEx, pd->d_Ex);
    t_ey = CheckGPUArraySilent(pd->dbgEy, pd->d_Ey);
    t_ez = CheckGPUArraySilent(pd->dbgEz, pd->d_Ez);
    t_hx = CheckGPUArraySilent(pd->dbgHx, pd->d_Hx);
    t_hy = CheckGPUArraySilent(pd->dbgHy, pd->d_Hy);
    t_hz = CheckGPUArraySilent(pd->dbgHz, pd->d_Hz);
    t_qx = CheckGPUArraySilent(pd->dbg_Qx, pd->d_Qx);
    t_qy = CheckGPUArraySilent(pd->dbg_Qy, pd->d_Qy);
    t_qz = CheckGPUArraySilent(pd->dbg_Qz, pd->d_Qz);

    if (num >= 500) {
        char wh[100];

        sprintf(wh, "%d", num);

        t_jx = checkGPUArray(pd->dbgJx, pd->d_Jx, "Jx", wh, nt);
        t_jy = checkGPUArray(pd->dbgJy, pd->d_Jy, "Jy", wh, nt);
        t_jz = checkGPUArray(pd->dbgJz, pd->d_Jz, "Jz", wh, nt);

        t_ex = checkGPUArray(pd->dbgEx, pd->d_Ex, "Ex", wh, nt);
        t_ey = checkGPUArray(pd->dbgEy, pd->d_Ey, "Ey", wh, nt);
        t_ez = checkGPUArray(pd->dbgEz, pd->d_Ez, "Ez", wh, nt);

    } else {

        t_ex = CheckGPUArraySilent(pd->dbgEx, pd->d_Ex);
        t_ey = CheckGPUArraySilent(pd->dbgEy, pd->d_Ey);
        t_ez = CheckGPUArraySilent(pd->dbgEz, pd->d_Ez);
    }

    t_jx = CheckGPUArraySilent(pd->dbgJx, pd->d_Jx);
    t_jy = CheckGPUArraySilent(pd->dbgJy, pd->d_Jy);
    t_jz = CheckGPUArraySilent(pd->dbgJz, pd->d_Jz);

    memory_monitor("checkControlPoint4", nt);

    double t_cmp_jx = checkGPUArray(pd->dbgJx, pd->d_Jx, "Jx", "step", nt);
    double t_cmp_jy = checkGPUArray(pd->dbgJy, pd->d_Jy, "Jy", "step", nt);
    double t_cmp_jz = checkGPUArray(pd->dbgJz, pd->d_Jz, "Jz", "step", nt);

#ifdef CONTROL_DIFF_GPU_PRINTS
    printf("GPU: Ex %15.5e Ey %15.5e Ez %15.5e \n",t_ex,t_ey,t_ez);
    printf("GPU: Hx %15.5e Hy %15.5e Ez %15.5e \n",t_hx,t_hy,t_hz);
    printf("GPU: Jx %15.5e Jy %15.5e Jz %15.5e \n",t_jx,t_jy,t_jz);
    printf("GPU compare : Jx %15.5e Jy %15.5e Jz %15.5e \n",t_cmp_jx,t_cmp_jy,t_cmp_jz);
#endif

    memory_monitor("checkControlPoint5", nt);

    double cp = checkControlPointParticles(num, pd->checkFile, nt);
    cout << "STEP: " <<  nt << endl;
    pd->f_prec_report = fopen("control_points.dat", "at");
    fprintf(pd->f_prec_report,
            " nt %5d\n num %6d\n Ex %15.5e\n Ey %15.5e\n Ez %15.5e\n Hx %15.5e\n Hy %15.5e\n Hz %15.5e\n Jx %15.5e\n Jy %15.5e\n Jz %15.5e\n Qx %15.5e\n Qy %15.5e\n Qz %15.5e\n particles %15.5e\n",
            nt, num,
            t_ex, t_ey, t_ez,
            t_hx, t_hy, t_hz,
            t_jx, t_jy, t_jz,
            t_qx, t_qy, t_qz,
            cp
    );

    fclose(pd->f_prec_report);

    memory_monitor("checkControlPoint6", nt);
#endif
}

double Plasma::CheckGPUArraySilent(double *a, double *d_a) {
    int err;
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;

    err = MemoryCopy(this->temp, d_a, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    CHECK_ERROR("MEM COPY", err);

    return CheckArraySilent(a, this->temp, (Nx + 2) * (Ny + 2) * (Nz + 2));
}

int Plasma::SetCurrentArraysToZero() {
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;
    int err;

    memset(pd->Jx, 0, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2));
    memset(pd->Jy, 0, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2));
    memset(pd->Jz, 0, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2));

    err = cudaMemset(pd->d_Jx, 0, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2));
    CHECK_ERROR("MEM SET", err);

    err = cudaMemset(pd->d_Jy, 0, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2));
    CHECK_ERROR("MEM SET", err);

    err = cudaMemset(pd->d_Jz, 0, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2));
    CHECK_ERROR("MEM SET", err);

    return 0;
}

int Plasma::SetCurrentsInCellsToZero() {
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;

    dim3 dimGrid((unsigned int)(Nx + 2), (unsigned int)(Ny + 2), (unsigned int)(Nz + 2)), dimBlockExt(CellExtent, CellExtent, CellExtent);

    void *args[] = {(void *) &pd->d_CellArray, 0};
    cudaLaunchKernel(
            (const void *) GPU_SetAllCurrentsToZero, // pointer to kernel func.
            dimGrid,                                 // grid
            dimBlockExt,                             // block
            args,                                    // arguments
            16000,
            0
    );

    return 0;
}

int Plasma::StepAllCells_fore_diagnostic(int nt) {
    memory_monitor("CellOrder_StepAllCells3", nt);

    return 0;
}

int Plasma::StepAllCells(int nt) {
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;
    dim3 dimGrid((unsigned int)(Nx + 2), (unsigned int)(Ny + 2), (unsigned int)(Nz + 2)), dimBlock(512, 1, 1);

    int err = cudaDeviceSynchronize();
    CHECK_ERROR("cudaDeviceSynchronize", err);

    std::cout << "begin step" << std::endl;

    void *args[] = {(void *) &pd->d_CellArray, 0};

    err = cudaLaunchKernel(
            (const void *) GPU_StepAllCells, // pointer to kernel func.
            dimGrid,                         // grid
            dimBlock,                        // block
            args,                            // arguments
            16000,
            0
    );
    CHECK_ERROR("GPU_StepAllCells", err);

    void *args1[] = {(void *) &pd->d_CellArray, &nt, 0};
    err = cudaFuncSetCacheConfig((const void *) GPU_CurrentsAllCells, cudaFuncCachePreferShared);
    CHECK_ERROR("cudaFuncSetCacheConfig", err);

    err = cudaLaunchKernel(
            (const void *) GPU_CurrentsAllCells, // pointer to kernel func.
            dimGrid,                             // grid
            dimBlock,                            // block
            args1,                               // arguments
            4000,
            0
    );

    CHECK_ERROR("GPU_CurrentsAllCells", err);

    std::cout << "end step" << std::endl;

    err = cudaDeviceSynchronize();
    CHECK_ERROR("cudaDeviceSynchronize", err);

    std::cout << "end step-12" << std::endl;

    return 0;
}

void Plasma::StepAllCells_post_diagnostic(int nt) {
    memory_monitor("CellOrder_StepAllCells4", nt);
}

int Plasma::WriteCurrentsFromCellsToArrays(int nt) {
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;
    dim3 dimGrid((unsigned int)(Nx + 2), (unsigned int)(Ny + 2), (unsigned int)(Nz + 2));
    dim3 dimExt(CellExtent, CellExtent, CellExtent);

    int zero = 0;
    void *args[] = {(void *) &pd->d_CellArray,
                    (void *) &zero,
                    (void *) &pd->d_Jx,
                    (void *) &pd->d_Jy,
                    (void *) &pd->d_Jz,
                    (void *) &pd->d_Rho,
                    0};

    cudaLaunchKernel(
            (const void *) GPU_WriteAllCurrents, // pointer to kernel func.
            dimGrid,                             // grid
            dimExt,                              // block
            args,                                // arguments
            16000,
            0
    );

    memory_monitor("CellOrder_StepAllCells5", nt);

    return 0;
}

int Plasma::MakeParticleList(int nt, int *stage, int **d_stage, int **d_stage1) {
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;
    int err;
    dim3 dimGrid((unsigned int)(Nx + 2), (unsigned int)(Ny + 2), (unsigned int)(Nz + 2)), dimBlockOne(1, 1, 1);

    err = cudaMalloc((void **) d_stage, sizeof(int) * (Nx + 2) * (Ny + 2) * (Nz + 2));
    CHECK_ERROR("MEM ALLOCATION", err);

    err = cudaMalloc((void **) d_stage1, sizeof(int) * (Nx + 2) * (Ny + 2) * (Nz + 2));
    CHECK_ERROR("MEM ALLOCATION", err);

    void *args[] = {
            (void *) &pd->d_CellArray,
            (void *) &nt,
            (void *) d_stage,
            0};

    err = cudaLaunchKernel(
            (const void *) GPU_MakeDepartureLists, // pointer to kernel func.
            dimGrid,                               // grid
            dimBlockOne,                           // block
            args,                                  // arguments
            16000,
            0
    );

    CHECK_ERROR("Launch kernel", err);

    err = cudaDeviceSynchronize();
    CHECK_ERROR("DEVICE SYNC", err);

    err = MemoryCopy(stage, *d_stage, sizeof(int) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    CHECK_ERROR("MEM COPY", err);

    return err;
}

int Plasma::inter_stage_diagnostic(int *stage, int nt) {
    if (stage[0] == TOO_MANY_PARTICLES) {
        printf("too many particles flying to (%d,%d,%d) from (%d,%d,%d) \n", stage[1], stage[2], stage[3], stage[4], stage[5], stage[6]);
        exit(0);
    }

    return 0;
}

int Plasma::reallyPassParticlesToAnotherCells(int nt, int *stage1, int *d_stage1) {
    int err;
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;

    dim3 dimGridBulk((unsigned int)Nx, (unsigned int)Ny, (unsigned int)Nz), dimBlockOne(1, 1, 1);
    cudaMemset(d_stage1, 0, sizeof(int) * (Nx + 2) * (Ny + 2) * (Nz + 2));

    void *args[] = {
            (void *) &pd->d_CellArray,
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
    CHECK_ERROR("GPU_ArrangeFlights", cudaStatus);

    err = MemoryCopy(stage1, d_stage1, sizeof(int) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    CHECK_ERROR("MEM COPY", err);

    memory_monitor("CellOrder_StepAllCells7", nt);

    return err;
}

int Plasma::reorder_particles(int nt) {
    int stage[4000], stage1[4000], *d_stage, *d_stage1, err;

    MakeParticleList(nt, stage, &d_stage, &d_stage1);

    inter_stage_diagnostic(stage, nt);

    err = reallyPassParticlesToAnotherCells(nt, stage1, d_stage1);

    return err;
}

void Plasma::Push(int nt) {
    StepAllCells_fore_diagnostic(nt);

    StepAllCells(nt);
    std::cout << "after StepAllCell" << std::endl;

    StepAllCells_post_diagnostic(nt);
}

int Plasma::SetCurrentsToZero() {
    SetCurrentArraysToZero();

    return SetCurrentsInCellsToZero();
}

void Plasma::CellOrder_StepAllCells(int nt) {
    SetCurrentsToZero();

    Push(nt);

    std::cout << "Push" << std::endl;

    WriteCurrentsFromCellsToArrays(nt);

    std::cout << "writeCut2arr" << std::endl;

    reorder_particles(nt);
}

double Plasma::checkControlPointParticlesOneSort(int check_point_num, const char * filename, GPUCell **copy_cells, int nt, int sort) {

    double t = 0.0;
    int size = 1;
#ifdef DEBUG
    double q_m, m;

    memory_monitor("checkControlPointParticlesOneSort", nt);

    Cell c0 = (*pd->AllCells)[0];

    readParticleParamsOneSort(filename, &pd->total_particles, &q_m, &m, sort);

    pd->dbg_y = new double[pd->total_particles];
    pd->dbg_x = new double[pd->total_particles];
    pd->dbg_z = new double[pd->total_particles];

    pd->dbg_px = new double[pd->total_particles];
    pd->dbg_py = new double[pd->total_particles];
    pd->dbg_pz = new double[pd->total_particles];

    readBinaryParticleArraysOneSort(filename, pd->dbg_x, pd->dbg_y, pd->dbg_z, pd->dbg_px, pd->dbg_py, pd->dbg_pz, pd->total_particles, nt, sort);

    memory_monitor("checkControlPointParticlesOneSort2", nt);

    size = (*pd->AllCells).size();

    for (int i = 0; i < size; i++) {
        GPUCell c = *(copy_cells[i]);

        t += c.checkCellParticles(check_point_num, pd->dbg_x, pd->dbg_y, pd->dbg_z, pd->dbg_px, pd->dbg_py, pd->dbg_pz, q_m, m);
    }

    memory_monitor("checkControlPointParticlesOneSort3", nt);

    delete[] pd->dbg_x;
    delete[] pd->dbg_y;
    delete[] pd->dbg_z;

    delete[] pd->dbg_px;
    delete[] pd->dbg_py;
    delete[] pd->dbg_pz;

    memory_monitor("checkControlPointParticlesOneSort4", nt);
#endif
    return t / size;
}

double Plasma::checkControlPointParticles(int check_point_num, const char * filename, int nt) {
    double te = 0.0, ti = 0.0, tb = 0.0;

#ifdef DEBUG
    copyCells(nt);

    memory_monitor("checkControlPointParticles", nt);

    ti = checkControlPointParticlesOneSort(check_point_num, filename, pd->cp, nt, ION);

    memory_monitor("checkControlPointParticles1", nt);

    te = checkControlPointParticlesOneSort(check_point_num, filename, pd->cp, nt, PLASMA_ELECTRON);

    memory_monitor("checkControlPointParticles1.5", nt);

    tb = checkControlPointParticlesOneSort(check_point_num, filename, pd->cp, nt, BEAM_ELECTRON);

    memory_monitor("checkControlPointParticles2", nt);

#endif

    return (te + ti + tb) / 3.0;
}

int Plasma::memory_monitor(std::string legend, int nt) {
#ifdef DEBUG
    size_t m_free, m_total;
    struct sysinfo info;

    int err = cudaMemGetInfo(&m_free, &m_total);
    CHECK_ERROR("cudaMemGetInfo", err);

    sysinfo(&info);
    printf("step %10d %50s GPU memory total %10d MB | free %10d MB | free CPU memory %10u MB\n", nt, legend.c_str(), (int) (m_total / 1024 / 1024), (int) (m_free / 1024 / 1024), (int) (info.freeram / 1024 / 1024));
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
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;

    NcFile dataFile(filename, NcFile::replace);
    dataFile.close();

    // copy dimensions
    NetCDFManipulator::plsm_add_dimension(filename.c_str(), "x", NX);
    NetCDFManipulator::plsm_add_dimension(filename.c_str(), "y", NY);
    NetCDFManipulator::plsm_add_dimension(filename.c_str(), "z", NZ);

    int err;

    err = MemoryCopy(this->temp, pd->d_Ex, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    CHECK_ERROR("MEM COPY", err);
    writeOne3DArray(filename.c_str(), this->temp, ELECTRIC_FIELD_LABEL + X_LABEL, UNITS_ELECTRIC_FIELD, DESC_ELECTRIC_FIELD + ELECTRIC_FIELD_LABEL + X_LABEL);
    err = MemoryCopy(this->temp, pd->d_Ey, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    CHECK_ERROR("MEM COPY", err);
    writeOne3DArray(filename.c_str(), this->temp, ELECTRIC_FIELD_LABEL + Y_LABEL, UNITS_ELECTRIC_FIELD, DESC_ELECTRIC_FIELD + ELECTRIC_FIELD_LABEL + Y_LABEL);
    err = MemoryCopy(this->temp, pd->d_Ez, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    CHECK_ERROR("MEM COPY", err);
    writeOne3DArray(filename.c_str(), this->temp, ELECTRIC_FIELD_LABEL + Z_LABEL, UNITS_ELECTRIC_FIELD, DESC_ELECTRIC_FIELD + ELECTRIC_FIELD_LABEL + Z_LABEL);

    err = MemoryCopy(this->temp, pd->d_Hx, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    CHECK_ERROR("MEM COPY", err);
    writeOne3DArray(filename.c_str(), this->temp, MAGNETIC_FIELD_LABEL + X_LABEL, UNITS_MAGNETIC_FIELD, DESC_MAGNETIC_FIELD + MAGNETIC_FIELD_LABEL + X_LABEL);
    err = MemoryCopy(this->temp, pd->d_Hy, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    CHECK_ERROR("MEM COPY", err);
    writeOne3DArray(filename.c_str(), this->temp, MAGNETIC_FIELD_LABEL + Y_LABEL, UNITS_MAGNETIC_FIELD, DESC_MAGNETIC_FIELD + MAGNETIC_FIELD_LABEL + Y_LABEL);
    err = MemoryCopy(this->temp, pd->d_Hz, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    CHECK_ERROR("MEM COPY", err);
    writeOne3DArray(filename.c_str(), this->temp, MAGNETIC_FIELD_LABEL + Z_LABEL, UNITS_MAGNETIC_FIELD, DESC_MAGNETIC_FIELD + MAGNETIC_FIELD_LABEL + Z_LABEL);

    err = MemoryCopy(this->temp, pd->d_Jx, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    CHECK_ERROR("MEM COPY", err);
    writeOne3DArray(filename.c_str(), this->temp, CURRENT_FIELD_LABEL + X_LABEL, UNITS_NO, CURRENT + CURRENT_FIELD_LABEL + X_LABEL);
    err = MemoryCopy(this->temp, pd->d_Jy, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    CHECK_ERROR("MEM COPY", err);
    writeOne3DArray(filename.c_str(), this->temp, CURRENT_FIELD_LABEL + Y_LABEL, UNITS_NO, CURRENT + CURRENT_FIELD_LABEL + Y_LABEL);
    err = MemoryCopy(this->temp, pd->d_Jz, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    CHECK_ERROR("MEM COPY", err);
    writeOne3DArray(filename.c_str(), this->temp, CURRENT_FIELD_LABEL + Z_LABEL, UNITS_NO, CURRENT + CURRENT_FIELD_LABEL + Z_LABEL);

    err = MemoryCopy(this->temp, pd->d_Qx, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    CHECK_ERROR("MEM COPY", err);
    writeOne3DArray(filename.c_str(), this->temp, MAGNETIC_HALF_STEP_FIELD_LABEL + X_LABEL, UNITS_NO, DESC_HALFSTEP + MAGNETIC_HALF_STEP_FIELD_LABEL + X_LABEL);
    err = MemoryCopy(this->temp, pd->d_Qy, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    CHECK_ERROR("MEM COPY", err);
    writeOne3DArray(filename.c_str(), this->temp, MAGNETIC_HALF_STEP_FIELD_LABEL + Y_LABEL, UNITS_NO, DESC_HALFSTEP + MAGNETIC_HALF_STEP_FIELD_LABEL + Y_LABEL);
    err = MemoryCopy(this->temp, pd->d_Qz, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), DEVICE_TO_HOST);
    CHECK_ERROR("MEM COPY", err);
    writeOne3DArray(filename.c_str(), this->temp, MAGNETIC_HALF_STEP_FIELD_LABEL + Z_LABEL, UNITS_NO, DESC_HALFSTEP + MAGNETIC_HALF_STEP_FIELD_LABEL + Z_LABEL);

}

/**
 * @param step {int} - step number
 */
void Plasma::Step(int step) {
    int Nx = pd->nx, Ny = pd->ny, Nz = pd->nz;

    ComputeField_FirstHalfStep(step);

    PushParticles(step);

    cout << "push ended" << endl;

    ComputeField_SecondHalfStep(step);
    cout << "field computed-2" << endl;

    sumMPI((Nx + 2) * (Ny + 2) * (Nz + 2), pd->d_Jx, pd->d_Jy, pd->d_Jz);

    Diagnose(step);
}

int Plasma::Compute() {
    cout << "----------------------------------------------------------- " << endl;
    size_t m_free, m_total;

    cudaMemGetInfo(&m_free, &m_total);

    if (pd->st <= 0 || pd->lst <= 0) {
        cout << "Invalid computation parameters values!" << endl;
        return 1;
    }

    for (int step = pd->st; step <= pd->lst; step++) {
        memory_monitor("before step", step);

        Step(step);

        // save file
        if ((pd->startSave <= step) && pd->saveStep > 0 && ((step - pd->startSave) % pd->saveStep == 0)) {
            writeDataToFile(step);
        }

        memory_monitor("after step", step);

        cout << "step " << step << " ===================" << endl;
    }

    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;

    return 0;
}

void Plasma::Initialize() {
    pi = new PlasmaInitializer(pd);
    pi->Initialize();
}