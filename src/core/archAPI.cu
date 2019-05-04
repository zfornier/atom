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

#ifdef __CUDACC__
int memory_monitor(std::string legend) {
#ifdef DEBUG
    size_t m_free, m_total;
    struct sysinfo info;

    int err = cudaMemGetInfo(&m_free, &m_total);
    CHECK_ERROR("cudaMemGetInfo", err);

    sysinfo(&info);
    printf("%40s | GPU memory: total %8d MB | free %8d MB | CPU memory: total %8u MB | free %8u MB\n",
           legend.c_str(),
           (int) (m_total / 1024 / 1024),
           (int) (m_free / 1024 / 1024),
           (int) (info.totalram / 1024 / 1024),
           (int) (info.freeram / 1024 / 1024));
#endif

    return 0;
}
#else
int memory_monitor(std::string legend) {
    return 0;
}
#endif