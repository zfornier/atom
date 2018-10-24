#ifndef ARCHAPI_H_
#define ARCHAPI_H_

#define HOST_TO_DEVICE   -131313
#define HOST_TO_HOST     -131314
#define DEVICE_TO_HOST   -131315
#define DEVICE_TO_DEVICE -131316

#define CHECK_ERROR(msg, err) \
                        if ((err) != 0) { \
                            std::cerr << (msg) << ": code: " << (err) << " : " << __FILE__ << " : " << __LINE__ << " : " << getErrorString((err)) << std::endl; \
                            exit(-1); \
                        }

#ifndef __CUDACC__

typedef struct {
    unsigned int x, y, z;
} uint3;

struct dim3 {
    unsigned int x, y, z;
#if defined(__cplusplus)
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
#endif /* __cplusplus */
};

typedef struct dim3 dim3;

#define __device__
#define __host__
#define __global__
#define __shared__

#endif

const char *getErrorString(int err);

int MemoryCopy(void* dst,void *src,size_t size,int dir);

int MemoryAllocate(void** dst,size_t size);

int getLastError();

#endif /* ARCHAPI_H_ */
