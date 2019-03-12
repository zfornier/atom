#ifndef ARCHAPI_H_
#define ARCHAPI_H_

#include <string.h>

#include "read_particles.h"

#define HOST_TO_DEVICE   -131313
#define HOST_TO_HOST     -131314
#define DEVICE_TO_HOST   -131315
#define DEVICE_TO_DEVICE -131316

#ifdef __CUDACC__

#define hostdevice_for_CUDA __host__ __device__
#define global_for_CUDA __global__
#endif

#ifndef __CUDACC__

typedef struct {
    double x, y, z;
} double3;

typedef struct {
    int x, y, z;
} int3;

typedef struct {
    unsigned int x, y, z;
} uint3;

struct dim3 {
    unsigned int x, y, z;
#if defined(__cplusplus)
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
#endif /* __cplusplus */
};

typedef struct dim3 dim3;

#define __device__
#define __host__
#define __global__
#define __shared__

#define OMP_NUM_THREADS 100

extern uint3 threadIdx, blockIdx;
#endif

const char *getErrorString(int err);

int MemoryCopy(void* dst,void *src,size_t size,int dir);

int MemoryAllocate(void** dst,size_t size);

#ifdef __CUDACC__

 int getLastError();

#else
 int getLastError();

#endif

#endif /* ARCHAPI_H_ */
