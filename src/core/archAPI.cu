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

    int status = (int)cudaMemcpy(dst,src,size,cuda_dir);

    return status != 0 ? -1 : 0;
}
#else

int MemoryCopy(void *dst, void *src, size_t size, int dir) {
    return 0;
}

#endif

#ifdef __CUDACC__
int MemoryAllocate(void** dst,size_t size) {
   int err = cudaMalloc(dst, size);

   return err != 0 ? -1 : 0;
}
#else
int MemoryAllocate(void **dst, size_t size) {
    return 0;
}
#endif

#ifdef __CUDACC__
int getLastError() {
    return (int)cudaGetLastError();
}
#else
int getLastError() {
    return 0;
}
#endif

#ifdef __CUDACC__
const char *getErrorString(int err) {
   return cudaGetErrorString((cudaError_t)err);
}
#else
const char *getErrorString(int err) {
    return "";
}
#endif
