/*
 * archAPI.cxx
 *
 *  Created on: Apr 10, 2018
 *      Author: snytav
 */

#include<stdlib.h>
#include<string.h>

#include "../include/archAPI.h"

#ifdef __CUDACC__
int MemoryCopy(void* dst,void *src,size_t size,int dir) {
   cudaMemcpyKind cuda_dir;

   if(dir == HOST_TO_DEVICE) cuda_dir = cudaMemcpyHostToDevice;
   if(dir == HOST_TO_HOST) cuda_dir = cudaMemcpyHostToHost;
   if(dir == DEVICE_TO_HOST) cuda_dir = cudaMemcpyDeviceToHost;
   if(dir == DEVICE_TO_DEVICE) cuda_dir = cudaMemcpyDeviceToDevice;

   return ((int)cudaMemcpy(dst,src,size,cuda_dir));
}
#else

int MemoryCopy(void *dst, void *src, size_t size, int dir);

#endif

#ifdef __CUDACC__
int MemoryAllocate(void** dst,size_t size) {
   cudaMalloc(dst,size);
   return 0;
}
#else

int MemoryAllocate(void **dst, size_t size);

#endif

#ifndef __CUDACC__

int getLastError() {
    return 0;
}

#else
int getLastError() {
    return (int)cudaGetLastError();
}
#endif

#ifdef __CUDACC__
const char *getErrorString(int err) {
   return cudaGetErrorString((cudaError_t)err);
}
#else

const char *getErrorString(int err) { return ""; }

#endif

int get_num_args(void **args) {
    int i;
    for (i = 0; args[i] != NULL; i++);

    return i;
}

#ifndef __CUDACC__
dim3 threadIdx, blockIdx;
#endif