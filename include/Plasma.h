#ifndef ATOM_PLASMA_H_
#define ATOM_PLASMA_H_

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <string>

#include "load_data.h"

//#include <unistd.h>
//#include <stdio.h>
#include <errno.h>

#ifdef __CUDACC__
#include <nvToolsExtCuda.h>
#include <nvToolsExtCudaRt.h>
#endif

#include "maxwell.h"

#include <time.h>

//#ifdef __OMP__
#include <omp.h>
//#endif

#ifdef __CUDACC__
#include <cuda.h>
#endif

#include "archAPI.h"
#include "maxwell.h"
#include "gpucell.h"
#include "mpi_shortcut.h"

#include "service_functions.h"

#include <sys/resource.h>
#include <stdint.h>

#include <sys/sysinfo.h>
#include <sys/time.h>


#include "init.h"
#include "diagnose.h"

#include <string>
#include <iostream>

#include "particle_target.h"

#include "memory_control.h"

#include "../src/add.cu"
#include "../src/wrap_kernel.cu"
//#include "run_control.h"
#include "../src/kernels.cu"
#include "../src/utils/NetCdf/read_file.cpp"
#include "../src/utils/NetCdf/write_file.cpp"

#include <vector>

using namespace std;

#define FORTRAN_ORDER

class Plasma {
public:
    GPUCell **h_CellArray, **d_CellArray;
    GPUCell **cp;
    thrust::device_vector <GPUCell> *d_AllCells;
    double *d_Ex, *d_Ey, *d_Ez, *d_Hx, *d_Hy, *d_Hz, *d_Jx, *d_Jy, *d_Jz, *d_Rho, *d_npJx, *d_npJy, *d_npJz;
    double *d_Qx, *d_Qy, *d_Qz;
    double *dbg_x, *dbg_y, *dbg_z, *dbg_px, *dbg_py, *dbg_pz;
    int total_particles;
    int h_controlParticleNumberArray[4000];
    int jx_wrong_points_number;
    int3 *jx_wrong_points, *d_jx_wrong_points;
//#ifdef ATTRIBUTES_CHECK
    double *ctrlParticles, *d_ctrlParticles, *check_ctrlParticles;
//#endif
    int jmp, size_ctrlParticles;
    double ami, amb, amf;
    int real_number_of_particle[3];
    FILE *f_prec_report;
    int CPU_field;
    int Nx, Ny, Nz;
    int n_per_cell;
    int meh;
    int magf;
    double ion_q_m, tau;
    double Lx, Ly, Lz;
    double ni;
    double *Qx, *Qy, *Qz, *dbg_Qx, *dbg_Qy, *dbg_Qz;
    double *Ex, *Ey, *Ez, *Hx, *Hy, *Hz, *Jx, *Jy, *Jz, *Rho, *npJx, *npJy, *npJz;
    double *dbgEx, *dbgEy, *dbgEz, *dbgHx, *dbgHy, *dbgHz, *dbgJx, *dbgJy, *dbgJz;
    double *dbgEx0, *dbgEy0, *dbgEz0;
    double *npEx, *npEy, *npEz;
    std::vector <GPUCell> *AllCells;

private:
    string dataFileStartPattern;
    string dataFileEndPattern;

public:
    Plasma(int, int, int, double, double, double, double, int, double, double);

    virtual ~Plasma();

    void copyCells(std::string, int);

    double checkGPUArray(double *, double *, std::string, std::string, int);

    double checkGPUArray(double *, double *);

    void virtual emeGPUIterate(int3, int3, double *, double *, double *, double *, double, double, double, int3, int3);

    void GetElectricFieldStartsDirs(int3 *, int3 *, int3 *, int);

    int virtual ElectricFieldTrace(double *, double *, double *, double *, int, double, double, double);

    int checkFields_beforeMagneticStageOne(double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, int);

    int checkFields_afterMagneticStageOne(double *, double *, double *, double *, double *, double *, double *, int);

    void ComputeField_FirstHalfStep(int);

    virtual void ComputeField_SecondHalfStep(int);

    void ElectricFieldComponentEvaluateTrace(
            double *, double *, double *, double *,
            int, double, double, double,
            int, int, int, int, int, int,
            int, int, int, int, int, int);

    void ElectricFieldComponentEvaluatePeriodic(
            double *, double *, double *, double *,
            int, double, double, double,
            int, int, int, int, int, int,
            int, int, int, int, int, int);

    void ElectricFieldEvaluate(double *, double *, double *, int, double *, double *, double *, double *, double *, double *);

    double3 getMagneticFieldTimeMeshFactors();

    virtual void MagneticStageOne(double *, double *, double *, double *, double *, double *, double *, double *, double *);

    virtual void MagneticFieldStageTwo(double *, double *, double *Hz, int, double *, double *, double *);

    int PushParticles(int);

    int readStartPoint(int);

    void Step(int);

    virtual double getElectricEnergy();

    void Diagnose(int);

    int getBoundaryLimit(int);

#include "../src/init.cu"

    int getMagneticFieldTraceShifts(int, int3 &, int3 &);

    int MagneticFieldTrace(double *, double *, double *, double *, int, int, int, double, double, int);

    int SimpleMagneticFieldTrace(Cell &, double *, double *, int, int, int);

    int PeriodicBoundaries(double *, int, int, int, int, int, int);

    int SinglePeriodicBoundary(double *, int, int, int, int, int, int);

    int SetPeriodicCurrentComponent(GPUCell **, double *, int, int, int, int);

    void SetPeriodicCurrents(int);

    void InitQdebug(std::string, std::string, std::string);

    void AssignCellsToArraysGPU();

    void AssignCellsToArrays();

    void write3Darray(char *, double *);

    void write3D_GPUArray(char *, double *);

    void readControlPoint(FILE **, char *, int, int, int, int,
                          double *, double *, double *,
                          double *, double *, double *,
                          double *, double *, double *,
                          double *, double *, double *,
                          double *, double *, double *,
                          double *, double *, double *);

    double checkControlMatrix(char *, int, char *, double *);

    void checkCurrentControlPoint(int, int);

    void checkControlPoint(int, int, int);

    void copyCellCurrentsToDevice(CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *, CellDouble *);

    double CheckArray(double *, double *, FILE *);

    double CheckArray(double *, double *);

    double CheckGPUArraySilent(double *, double *);

    int CheckValue(double *, double *, int);

    void read3DarrayLog(char *, double *, int, int);

    void read3Darray(char *, double *);

    void read3Darray(string, double *);

    int PeriodicCurrentBoundaries(double *, int, int, int, int, int, int);

    void ClearAllParticles();

    int initControlPointFile();

    int copyCellsWithParticlesToGPU();

    void ListAllParticles(int, std::string);

    double TryCheckCurrent(int, double *);

    double checkNonPeriodicCurrents(int);

    double checkPeriodicCurrents(int);

    int SetCurrentArraysToZero();

    int SetCurrentsInCellsToZero(int);

    int StepAllCells_fore_diagnostic(int);

    int StepAllCells(int, double, double);

    int StepAllCells_post_diagnostic(int);

    int WriteCurrentsFromCellsToArrays(int);

    int MakeParticleList(int, int *, int *, int **, int **);

    int inter_stage_diagnostic(int *, int);

    int reallyPassParticlesToAnotherCells(int, int *, int *);

    int reorder_particles(int);

    int Push(int, double, double);

    int SetCurrentsToZero(int);

    void CellOrder_StepAllCells(int, double, double, int);

    double checkControlPointParticlesOneSort(int, FILE *, GPUCell **, int, int);

    double checkControlPointParticles(int, FILE *, char *, int);

    int readControlFile(int);

    int memory_monitor(std::string, int);

    int memory_status_print(int);

    void writeDataToFile(int);

    int Compute(int, int, int, int);

    int Compute(int, int);
};

#endif /* GPU_PLASMA_H_ */
