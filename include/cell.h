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
#include "archAPI.h"

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

    __host__ __device__
    int AllocParticles();

    __host__ __device__
    double ParticleArrayRead(int n_particle, int attribute);

    __host__ __device__
    void ParticleArrayWrite(int n_particle, int attribute, double t);

    __host__ __device__
    void writeParticleToSurface(int n, Particle *p);

    __host__ __device__
    void addParticleToSurface(Particle *p, int *number_of_particles);

    __host__ __device__
    Particle readParticleFromSurfaceDevice(int n);

    __host__ __device__
    void removeParticleFromSurfaceDevice(int n, Particle *p, int *number_of_particles);

    __host__ __device__
    double get_hx();

    __host__ __device__
    double get_hy();

    __host__ __device__
    double get_hz();

#ifdef GPU_PARTICLE
    double *GetParticles();
#else
    thrust::host_vector<Particle>&  GetParticles();
#endif

    __host__ __device__
    double getCellFraction(double x, double x0, double hx);

    __host__ __device__
    int getCellNumber(double x, double x0, double hx);

    __host__ __device__
    int getCellNumberCenter(double x, double x0, double hx);

    __host__ __device__
    int getCellNumberCenterCurrent(double x, double x0, double hx);

    __host__ __device__
    int getPointPosition(double x, double x0, double hx);

    __host__ __device__
    int getPointCell(double3 x);

    __host__ __device__
    double getCellTransitAverage(double hz, int i1, int i2, double x0);

    __host__ __device__
    double getCellTransitRatio(double z1, double z, double z2);

    __host__ __device__
    double getCellTransitProduct(double z1, double z, double z2);

    __host__ __device__
    double getRatioBasedX(double x1, double x, double s);

    __host__ __device__
    double getCenterRelatedShift(double x, double x1, int i, double hx, double x0);

    __host__ __device__
    void flyDirection(Particle *p, int *dx, int *dy, int *dz);

    __host__ __device__
    void inverseDirection(int *dx, int *dy, int *dz);

    __host__ __device__
    bool isPointInCell(double3 x);

    __host__ __device__
    Cell();

    __host__ __device__
    ~Cell();

    __host__ __device__
    Cell(int i1, int l1, int k1, double Lx, double Ly, double Lz, int Nx1, int Ny1, int Nz1, double tau1);

#ifdef VIRTUAL_FUNCTIONS
    virtual
#endif
    __host__ __device__
    double3 GetElectricField(
            int3 i, int3 i1,
            double &s1, double &s2, double &s3, double &s4, double &s5, double &s6,
            double &s11, double &s21, double &s31, double &s41, double &s51, double &s61,
            Particle *p, CellDouble *Ex1, CellDouble *Ey1, CellDouble *Ez1
    );

    __host__ __device__
    int3 getCellTripletNumber(int n);

    __host__ __device__
    int getGlobalCellNumberTriplet(int *i, int *l, int *k);

    __host__ __device__
    int getGlobalCellNumber(int i, int l, int k);

    __host__ __device__
    int getWrapCellNumberTriplet(int *i, int *l, int *k);

    __host__ __device__
    int getWrapCellNumber(int i, int l, int k);

    __host__ __device__
    int getFortranCellNumber(int i, int l, int k);

    __host__ __device__
    void getFortranCellTriplet(int n, int *i, int *l, int *k);

    __host__ __device__
    int getGlobalBoundaryCellNumber(int i, int k, int dir, int N);

    __host__ __device__
    void ClearParticles();

    __host__ __device__
    void Init();

    __host__ __device__
#ifdef VIRTUAL_FUNCTIONS
    virtual
#endif
    void InverseKernel(double3 x, int3 &i, int3 &i1,
                       double &s1, double &s2, double &s3, double &s4, double &s5, double &s6,
                       double &s11, double &s21, double &s31, double &s41, double &s51, double &s61,
                       Particle *p);

    __host__ __device__
    double Interpolate3D(CellDouble *E, int3 *cell, double sx, double sx1, double sy, double sy1, double sz, double sz1, Particle *p, int n);

    __host__ __device__
#ifdef VIRTUAL_FUNCTIONS
    virtual
#endif
    double3 GetMagneticField(
            int3 i, int3 i1,
            double &s1, double &s2, double &s3, double &s4, double &s5, double &s6,
            double &s11, double &s21, double &s31, double &s41, double &s51, double &s61,
            Particle *p, CellDouble *Hx1, CellDouble *Hy1, CellDouble *Hz1
    );

    __host__ __device__
    double s1_interpolate(double x);

    __host__ __device__
    double s2_interpolate(double x);

    __host__ __device__
    double s3_interpolate(double y);

    __host__ __device__
    double s5_interpolate(double z);

    __host__ __device__
    double s4_interpolate(double y);

    __host__ __device__
    double s6_interpolate(double z);

    __host__ __device__
    Field GetField(Particle *p, CellTotalField cf);

    __host__ __device__
    void CurrentToMesh(double tau, int *cells, DoubleCurrentTensor *dt, Particle *p);

    __host__ __device__
    void Reflect(Particle *p);

    __host__ __device__
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
    void pqr(int3 &i, double3 &x, double3 &x1, double &a1, double tau, CurrentTensor *t1, int num, Particle *p);

    __host__ __device__
#ifdef VIRTUAL_FUNCTIONS
    virtual
#endif
    void pqr(int3 &i, double3 &x, double3 &x1, double &a1, double tau);

    __host__ __device__
    bool Insert(Particle &p);

    __host__ __device__
#ifdef VIRTUAL_FUNCTIONS
    virtual
#endif
    void MoveSingleParticle(unsigned int i, CellTotalField cf);

    __host__ __device__
    DoubleCurrentTensor AccumulateCurrentSingleParticle(unsigned int i, int *cells, DoubleCurrentTensor *dt);

    __host__ __device__
    void SetAllCurrentsToZero(uint3 threadIdx);

    __host__ __device__
    double getFortranArrayValue(double *E, int i, int l, int k);

    __host__ __device__
    void readField(double *E, CellDouble &Ec, uint3 threadIdx);

    __host__ __device__
    void readField(double *E, CellDouble &Ec);

    __host__ __device__
    void readFieldsFromArrays(double *glob_Ex, double *glob_Ey, double *glob_Ez, double *glob_Hx, double *glob_Hy, double *glob_Hz, uint3 threadIdx);

    __host__ __device__
    void readFieldsFromArrays(double *glob_Ex, double *glob_Ey, double *glob_Ez, double *glob_Hx, double *glob_Hy, double *glob_Hz);

    __host__ __device__
    Cell &operator=(Cell const &src);

    double compareParticleLists(Cell *c);

    double
#ifdef VIRTUAL_FUNCTIONS
    virtual
#endif
    compareToCell(Cell &src);

    double checkCellParticles(int check_point_num, double *x, double *y, double *z, double *px, double *py, double *pz, double q_m, double m);

    void SetControlSystem(int j, double *c);

    __host__ __device__
    void SetControlSystemToParticles();
};

#endif
