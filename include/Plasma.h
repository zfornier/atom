#ifndef ATOM_PLASMA_H_
#define ATOM_PLASMA_H_

#ifdef __CUDACC__
#include <cuda.h>
#include <nvToolsExtCuda.h>
#include <nvToolsExtCudaRt.h>
#endif

#include "mpi_shortcut.h"
#include "service_functions.h"
#include "load_data.h"
#include "gpucell.h"
#include "kernels.h"
#include "NetCdf/read_file.h"
#include "PlasmaTypes.h"
#include "PlasmaInitializer.h"
#include "archAPI.h"
#include "maxwell.h"

using namespace std;

class Plasma {
private:
    PlasmaConfig * pd;
    PlasmaInitializer * pi;
    string dataFileStartPattern;
    string dataFileEndPattern;
    double * temp;

public:
    Plasma(PlasmaConfig * p);

    int Compute();

    void Initialize();

    virtual ~Plasma();

private:
    void copyCells();

    double CheckGPUArraySilent(double *, double *);

    void virtual emeGPUIterate(int3, int3, double *, double *, double *, double *, double, double, double, int3, int3);

    void GetElectricFieldStartsDirs(int3 *, int3 *, int3 *, int);

    int virtual ElectricFieldTrace(double *, double *, double *, double *, int, double, double, double);

    void ComputeField_FirstHalfStep();

    virtual void ComputeField_SecondHalfStep();

    void ElectricFieldComponentEvaluateTrace(double *, double *, double *, double *, int, double, double, double);

    void ElectricFieldComponentEvaluatePeriodic(double *, int, int, int, int, int, int, int, int, int, int, int, int, int);

    void ElectricFieldEvaluate(double *, double *, double *, double *, double *, double *, double *, double *, double *);

    double3 getMagneticFieldTimeMeshFactors();

    virtual void MagneticStageOne(double *, double *, double *, double *, double *, double *, double *, double *, double *);

    virtual void MagneticFieldStageTwo(double *, double *, double *Hz, double *, double *, double *);

    int PushParticles();

    void Step(int);

    virtual double getElectricEnergy();

    void Diagnose(int);

    int getMagneticFieldTraceShifts(int, int3 &, int3 &);

    int MagneticFieldTrace(double *, double *, double *, double *, int, int, int, double, double, int);

    int SimpleMagneticFieldTrace(Cell &, double *, double *, int, int, int);

    int PeriodicBoundaries(double *, int, int, int, int, int, int);

    int SetPeriodicCurrentComponent(GPUCell **, double *, int, unsigned int, unsigned int, unsigned int);

    void SetPeriodicCurrents();

    void AssignCellsToArraysGPU();

    void readControlPoint(const char *, int, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *);

    void checkControlPoint(int);

    int SetCurrentArraysToZero();

    int SetCurrentsInCellsToZero();

    int StepAllCells();

    int WriteCurrentsFromCellsToArrays();

    int MakeParticleList(int *, int **, int **);

    int reallyPassParticlesToAnotherCells(int *, int *);

    int reorder_particles();

    void Push();

    int SetCurrentsToZero();

    void CellOrder_StepAllCells();

    double checkControlPointParticlesOneSort(const char * , GPUCell **, int, int);

    double checkControlPointParticles(const char *, int);

    void writeDataToFile(int);

};

#endif /* GPU_PLASMA_H_ */
