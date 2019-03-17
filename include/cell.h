#ifndef CELL_H
#define CELL_H

//#define VIRTUAL_FUNCTIONS

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "particle.h"
#include "compare.h"

#define CellExtent 5
#define PARTICLE_MASS_TOLERANCE 1e-15
#define WRONG_PARTICLE_TYPE -13333
#define PARTICLES_FLYING_ONE_DIRECTION 50
#define TOO_MANY_PARTICLES -513131313

typedef struct CellDouble {
    double M[CellExtent][CellExtent][CellExtent];
} CellDouble;

typedef struct CellTotalField {
    CellDouble *Ex, *Ey, *Ez, *Hx, *Hy, *Hz;
} CellTotalField;

#define MAX_PPC   5000
const int MAX_particles_per_cell = MAX_PPC;

__host__ __device__ int isNan(double t);

//QUESTION: what was the reason to have class implementation in .h file?

class Cell {
public:
    int i, l, k;
    double hx, hy, hz, tau;
    double x0, y0, z0;
    double xm, ym, zm;
    int Nx, Ny, Nz;
    int jmp;
    double *d_ctrlParticles;

    CellDouble *Jx, *Ex, *Hx, *Jy, *Ey, *Hy, *Jz, *Ez, *Hz, *Rho;

#ifdef GPU_PARTICLE
    double *doubParticleArray;
    int number_of_particles;
    int busyParticleArray;
    int arrival[3][3][3], departure[3][3][3];
    int departureListLength;
    Particle departureList[3][3][3][PARTICLES_FLYING_ONE_DIRECTION];
    double *doubArrivalArray;
#else
    thrust::host_vector<Particle> all_particles;
#endif

//TODO: it's not an elegant way. Use the following approach:
/*
#ifdef __CUDACC__
    #define host_device __host__ __device__
#lese
    #define host_device 
#endif
    host_device int AllocParticles....
    host_device double ParticleArrayRead ....
*/
#ifdef __CUDACC__
    __host__ __device__
#endif
    int AllocParticles();

#ifdef __CUDACC__
    __host__ __device__
#endif
    double ParticleArrayRead(int n_particle, int attribute);

#ifdef __CUDACC__
    __host__ __device__
#endif
    void ParticleArrayWrite(int n_particle, int attribute, double t);

#ifdef __CUDACC__
    __host__ __device__
#endif
    void writeParticleToSurface(int n, Particle *p);

#ifdef __CUDACC__
    __host__ __device__
#endif
    void addParticleToSurface(Particle *p, int *number_of_particles);

#ifdef __CUDACC__
    __host__ __device__
#endif
    Particle readParticleFromSurfaceDevice(int n);

public:

#ifdef __CUDACC__
    __host__ __device__
#endif
    void removeParticleFromSurfaceDevice(int n, Particle *p, int *number_of_particles);

#ifdef __CUDACC__
    __host__ __device__
#endif
    double get_hx();

#ifdef __CUDACC__
    __host__ __device__
#endif
    double get_hy();

#ifdef __CUDACC__
    __host__ __device__
#endif
    double get_hz();

#ifdef GPU_PARTICLE

    double *GetParticles();

#else
    thrust::host_vector<Particle>&  GetParticles();
#endif

#ifdef __CUDACC__
    __host__ __device__
#endif
    double getCellFraction(double x, double x0, double hx);

#ifdef __CUDACC__
    __host__ __device__
#endif
    int getCellNumber(double x, double x0, double hx);

    __host__ __device__
#ifdef __CUDACC__
    __host__ __device__
#endif
    int getCellNumberCenter(double x, double x0, double hx);

#ifdef __CUDACC__
    __host__ __device__
#endif
    int getCellNumberCenterCurrent(double x, double x0, double hx);

#ifdef __CUDACC__
    __host__ __device__
#endif
    int getPointPosition(double x, double x0, double hx);

#ifdef __CUDACC__
    __host__ __device__
#endif
    int getPointCell(double3 x);

#ifdef __CUDACC__
    __host__ __device__
#endif
    double getCellReminder(double x, double x0, double hx);

#ifdef __CUDACC__
    __host__ __device__
#endif
    double getCellCenterReminder(double x, double x0, double hx);


#ifdef __CUDACC__
    __host__ __device__
#endif
    double getCellTransitAverage(double hz, int i1, int i2, double x0);


#ifdef __CUDACC__
    __host__ __device__
#endif
    double getCellTransitRatio(double z1, double z, double z2);


#ifdef __CUDACC__
    __host__ __device__
#endif
    double getCellTransitProduct(double z1, double z, double z2);


#ifdef __CUDACC__
    __host__ __device__
#endif
    double getRatioBasedX(double x1, double x, double s);


#ifdef __CUDACC__
    __host__ __device__
#endif
    double getCenterRelatedShift(double x, double x1, int i, double hx, double x0);

#ifdef __CUDACC__
    __host__ __device__
#endif
    void flyDirection(Particle *p, int *dx, int *dy, int *dz);

#ifdef __CUDACC__
    __host__ __device__
#endif
    __host__ __device__ void inverseDirection(int *dx, int *dy, int *dz);


#ifdef __CUDACC__
    __host__ __device__
#endif
    bool isPointInCell(double3 x);

    //public:
#ifdef __CUDACC__
    __host__ __device__
#endif

    Cell();

#ifdef __CUDACC__
    __host__ __device__
#endif

    ~Cell();

#ifdef __CUDACC__
    __host__ __device__
#endif

    Cell(int i1, int l1, int k1, double Lx, double Ly, double Lz, int Nx1, int Ny1, int Nz1, double tau1);

    __host__ __device__
#ifdef VIRTUAL_FUNCTIONS
    virtual
#endif
#ifdef __CUDACC__
    __host__ __device__
#endif
    double3 GetElectricField(
            int3 i, int3 i1,
            double &s1, double &s2, double &s3, double &s4, double &s5, double &s6,
            double &s11, double &s21, double &s31, double &s41, double &s51, double &s61,
            Particle *p, CellDouble *Ex1, CellDouble *Ey1, CellDouble *Ez1
    );

#ifdef __CUDACC__
    __host__ __device__
#endif
    int3 getCellTripletNumber(int n);

#ifdef __CUDACC__
    __host__ __device__
#endif
    int getGlobalCellNumberTriplet(int *i, int *l, int *k);

#ifdef __CUDACC__
    __host__ __device__
#endif
    int getGlobalCellNumber(int i, int l, int k);

#ifdef __CUDACC__
    __host__ __device__
#endif
    int getWrapCellNumberTriplet(int *i, int *l, int *k);

#ifdef __CUDACC__
    __host__ __device__
#endif
    int getWrapCellNumber(int i, int l, int k);

#ifdef __CUDACC__
    __host__ __device__
#endif
    int getFortranCellNumber(int i, int l, int k);

#ifdef __CUDACC__
    __host__ __device__
#endif
    void getFortranCellTriplet(int n, int *i, int *l, int *k);

#ifdef __CUDACC__
    __host__ __device__
#endif
    int getGlobalBoundaryCellNumber(int i, int k, int dir, int N);

#ifdef __CUDACC__
    __host__ __device__
#endif
    void ClearParticles();

#ifdef __CUDACC__
    __host__ __device__
#endif
    void Init();

#ifdef __CUDACC__
    __host__ __device__
#endif
#ifdef VIRTUAL_FUNCTIONS
    virtual
#endif
    void InverseKernel(double3 x, int3 &i, int3 &i1,
                       double &s1, double &s2, double &s3, double &s4, double &s5, double &s6,
                       double &s11, double &s21, double &s31, double &s41, double &s51, double &s61,
                       Particle *p
    );

#ifdef __CUDACC__
    __host__ __device__
#endif
    double Interpolate3D(CellDouble *E, int3 *cell, double sx, double sx1, double sy, double sy1, double sz, double sz1, Particle *p, int n);

#ifdef __CUDACC__
    __host__ __device__
#endif
#ifdef VIRTUAL_FUNCTIONS
    virtual
#endif
    double3 GetMagneticField(
            int3 i, int3 i1,
            double &s1, double &s2, double &s3, double &s4, double &s5, double &s6,
            double &s11, double &s21, double &s31, double &s41, double &s51, double &s61,
            Particle *p, CellDouble *Hx1, CellDouble *Hy1, CellDouble *Hz1
    );

#ifdef __CUDACC__
    __host__ __device__
#endif

    double s1_interpolate(double x);

#ifdef __CUDACC__
    __host__ __device__
#endif
    double s2_interpolate(double x);

#ifdef __CUDACC__
    __host__ __device__
#endif
    double s3_interpolate(double y);

#ifdef __CUDACC__
    __host__ __device__
#endif
    double s5_interpolate(double z);

#ifdef __CUDACC__
    __host__ __device__
#endif
    double s4_interpolate(double y);

#ifdef __CUDACC__
    __host__ __device__
#endif
    double s6_interpolate(double z);

#ifdef __CUDACC__
    __host__ __device__
#endif
    Field GetField(Particle *p, CellTotalField cf);

#ifdef __CUDACC__
    __host__ __device__
#endif
    void CurrentToMesh(double tau, int *cells, DoubleCurrentTensor *dt, Particle *p);

#ifdef __CUDACC__
    __host__ __device__
#endif
    void Reflect(Particle *p);

#ifdef __CUDACC__
    __host__ __device__
#endif
#ifdef VIRTUAL_FUNCTIONS
    virtual
#endif
    void Kernel(CellDouble &Jx,
                int i11, int i12, int i13,
                int i21, int i22, int i23,
                int i31, int i32, int i33,
                int i41, int i42, int i43,
                double su, double dy, double dz, double dy1, double dz1, double s1);

    __host__ __device__
#ifdef VIRTUAL_FUNCTIONS
    virtual
#endif

    void pqr(int3 &i, double3 &x, double3 &x1, double &a1, double tau, CurrentTensor *t1,
             int num, Particle *p);

#ifdef __CUDACC__
    __host__ __device__
#endif
#ifdef VIRTUAL_FUNCTIONS
    virtual
#endif
    void pqr(int3 &i, double3 &x, double3 &x1, double &a1, double tau);


#ifdef __CUDACC__
    __host__ __device__
#endif
    bool Insert(Particle &p);

#ifdef __CUDACC__
    __host__ __device__
#endif
#ifdef VIRTUAL_FUNCTIONS
    virtual
#endif
    void MoveSingleParticle(unsigned int i, CellTotalField cf);

#ifdef __CUDACC__
    __host__ __device__
#endif
    DoubleCurrentTensor AccumulateCurrentSingleParticle(unsigned int i, int *cells, DoubleCurrentTensor *dt);


#ifdef __CUDACC__
    __host__ __device__
#endif
    void SetAllCurrentsToZero(uint3 threadIdx);

#ifdef __CUDACC__
    __host__ __device__
#endif
    double getFortranArrayValue(double *E, int i, int l, int k);

// MAPPING: fORTRAN NODE i GOES TO 2nd NODE OF C++ CELL i-1
// Simple : C++ (i+i1) ----->>>> F(i+i1-1)
#ifdef __CUDACC__
    __host__ __device__
#endif
    void readField(double *E, CellDouble &Ec, uint3 threadIdx);

#ifdef __CUDACC__
    __host__ __device__
#endif
    void readField(double *E, CellDouble &Ec);

#ifdef __CUDACC__
    __host__ __device__
#endif
    void readFieldsFromArrays(double *glob_Ex, double *glob_Ey, double *glob_Ez, double *glob_Hx, double *glob_Hy,
                              double *glob_Hz, uint3 threadIdx);

#ifdef __CUDACC__
    __host__ __device__
#endif
    void readFieldsFromArrays(double *glob_Ex, double *glob_Ey, double *glob_Ez, double *glob_Hx, double *glob_Hy, double *glob_Hz);

#ifdef __CUDACC__
    __host__ __device__
#endif
    Cell &operator=(Cell const &src);

    double compareParticleLists(Cell *c);

    double
#ifdef VIRTUAL_FUNCTIONS
    virtual
#endif
    compareToCell(Cell &src);

    double checkCellParticles(int check_point_num, double *x, double *y, double *z, double *px, double *py, double *pz, double q_m, double m);

    void SetControlSystem(int j, double *c);

#ifdef __CUDACC__
    __host__ __device__
#endif
    void SetControlSystemToParticles();
};

#endif
