#ifndef ATOM_PLASMA_H_
#define ATOM_PLASMA_H_

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string>
#include <errno.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <nvToolsExtCuda.h>
#include <nvToolsExtCudaRt.h>
#endif

#include <sys/resource.h>
#include <stdint.h>
#include <sys/sysinfo.h>
#include <iostream>
#include <vector>

#include "mpi_shortcut.h"
#include "service_functions.h"
#include "load_data.h"
#include "gpucell.h"
#include "kernels.h"
#include "NetCdf/read_file.h"
#include "PlasmaTypes.h"
#include "PlasmaInitializer.h"

#define FORTRAN_ORDER

using namespace std;

class Plasma {
public:
    int Nx, Ny, Nz;
    double Lx, Ly, Lz;
    int n_per_cell;
    double ni;
    double tex0, tey0, tez0, Tb, rimp, rbd;
    double ion_q_m, tau;
    int jmp;

    int total_particles;
    int size_ctrlParticles;
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

private:
    PlasmaConfig * pd;

    string dataFileStartPattern;
    string dataFileEndPattern;

public:
    Plasma(int, int, int, double, double, double, double, int, double, double);

    Plasma(PlasmaConfig * p);

    int Compute(int, int, int, int);

    int Compute(int, int);

    void Initialize();

    virtual ~Plasma();

//private:
    void copyCells(std::string, int);

    double checkGPUArray(double *, double *, std::string, std::string, int);

    double checkGPUArray(double *, double *);

    void virtual emeGPUIterate(int3, int3, double *, double *, double *, double *, double, double, double, int3, int3);

    void GetElectricFieldStartsDirs(int3 *, int3 *, int3 *, int);

    int virtual ElectricFieldTrace(double *, double *, double *, double *, int, double, double, double);

    int checkFields_beforeMagneticStageOne(int);

    int checkFields_afterMagneticStageOne(int);

    void checkCudaError();

    void ComputeField_FirstHalfStep(int);

    virtual void ComputeField_SecondHalfStep(int);

    void ElectricFieldComponentEvaluateTrace(double *, double *, double *, double *, int, double, double, double);

    void ElectricFieldComponentEvaluatePeriodic(double *, int, int, int, int, int, int, int, int, int, int, int, int, int);

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
    
//TODO: It's an ugly style to include cu files. Include cuh (header files) only
//TODO: you need to include hardware specific files only if an appropriate compiler is used.
#include "../src/core/init.cu"

    int getMagneticFieldTraceShifts(int, int3 &, int3 &);

    int MagneticFieldTrace(double *, double *, double *, double *, int, int, int, double, double, int);

    int SimpleMagneticFieldTrace(Cell &, double *, double *, int, int, int);

    int PeriodicBoundaries(double *, int, int, int, int, int, int);

    int SetPeriodicCurrentComponent(GPUCell **, double *, int, int, int, int);

    void SetPeriodicCurrents(int);

    void AssignCellsToArraysGPU();

    void readControlPoint(FILE **, char *, int, int, int, int, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *);

    void checkControlPoint(int, int, int);

    double CheckArray(double *, double *, FILE *);

    double CheckArray(double *, double *);

    double CheckGPUArraySilent(double *, double *);

    void read3DarrayLog(char *, double *, int, int);

    void read3Darray(char *, double *);

    void read3Darray(string, double *);

    void ClearAllParticles();

    int initControlPointFile();

    int copyCellsWithParticlesToGPU();

    int SetCurrentArraysToZero();

    int SetCurrentsInCellsToZero(int);

    int StepAllCells_fore_diagnostic(int);

    int StepAllCells(int, double, double);

    void StepAllCells_post_diagnostic(int);

    int WriteCurrentsFromCellsToArrays(int);

    int MakeParticleList(int, int *, int **, int **);

    int inter_stage_diagnostic(int *, int);

    int reallyPassParticlesToAnotherCells(int, int *, int *);

    int reorder_particles(int);

    void Push(int, double, double);

    int SetCurrentsToZero(int);

    void CellOrder_StepAllCells(int, double, double, int);

    double checkControlPointParticlesOneSort(int, FILE *, GPUCell **, int, int);

    double checkControlPointParticles(int, FILE *, char *, int);

    int readControlFile(int);

    int memory_monitor(std::string, int);

    int memory_status_print(int);

    void writeDataToFile(int);

};

#endif /* GPU_PLASMA_H_ */
