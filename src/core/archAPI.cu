/*
 * archAPI.cxx
 *
 *  Created on: Apr 10, 2018
 *      Author: snytav
 */

#include "../../include/archAPI.h"

#ifdef __CUDACC__
int MemoryCopy(void* dst,void *src,size_t size,int dir) {
   cudaMemcpyKind cuda_dir;

   if(dir == HOST_TO_DEVICE) cuda_dir = cudaMemcpyHostToDevice;
   if(dir == HOST_TO_HOST) cuda_dir = cudaMemcpyHostToHost;
   if(dir == DEVICE_TO_HOST) cuda_dir = cudaMemcpyDeviceToHost;
   if(dir == DEVICE_TO_DEVICE) cuda_dir = cudaMemcpyDeviceToDevice;
// It's not correct! You can't return architecture specific result
// You must return Error or not_Error and in case of DEBUG  check the error here
   return ((int)cudaMemcpy(dst,src,size,cuda_dir));
}
#else

int MemoryCopy(void *dst, void *src, size_t size, int dir);

#endif

#ifdef __CUDACC__
int MemoryAllocate(void** dst,size_t size) {
// ERROR: cudaMalloc may failed. Check the error here and return it as it was describe in the comment above
   cudaMalloc(dst,size);
   return 0;
}
#else

int MemoryAllocate(void **dst, size_t size);

#endif

// ERROR: why it's ihndef here while you have ifdef otherwice?
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

#ifndef __CUDACC__
//dim3 threadIdx, blockIdx;
#endif
