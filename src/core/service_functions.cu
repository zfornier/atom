#include "../../include/service_functions.h"

using namespace std;

int setPrintfLimit() {
    size_t sizeP;

    std::cout << "Particle size " << sizeof(Particle) << " : " << sizeof(Particle) / sizeof(double) << ". CurrentTensor " << (int)sizeof(CurrentTensor) << " short " << (int)sizeof(char) << std::endl;

    cudaDeviceGetLimit(&sizeP, cudaLimitPrintfFifoSize);

    std::cout << "print default limit " << sizeP / 1024 / 1024 << std::endl;

    sizeP *= 10000;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sizeP);

    cudaDeviceGetLimit(&sizeP, cudaLimitPrintfFifoSize);

    std::cout << "print limit set to " << sizeP / 1024 / 1024 << std::endl;

    return 0;
}

double CheckArraySilent(double *a, double *dbg_a, int size) {
    double diff = 0.0;

    for (int n = 0; n < size; n++) {
        diff += pow(a[n] - dbg_a[n], 2.0);
    }

    return pow(diff / (size), 0.5);
}

void cudaMalloc3D(double **X, double **Y, double **Z, int nx, int ny, int nz) {
    int err;
    err = cudaMalloc((void **) X, sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));
    CHECK_ERROR("CUDA MALLOC", err);
    err = cudaMalloc((void **) Y, sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));
    CHECK_ERROR("CUDA MALLOC", err);
    err = cudaMalloc((void **) Z, sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));
    CHECK_ERROR("CUDA MALLOC", err);
}

void copyFieldsToGPU(
        double *d_Ex, double *d_Ey, double *d_Ez,
        double *d_Hx, double *d_Hy, double *d_Hz,
        double *d_Jx, double *d_Jy, double *d_Jz,
        double *d_npJx, double *d_npJy, double *d_npJz,
        double *d_Qx, double *d_Qy, double *d_Qz,
        double *Ex, double *Ey, double *Ez,
        double *Hx, double *Hy, double *Hz,
        double *Jx, double *Jy, double *Jz,
        double *npJx, double *npJy, double *npJz,
        double *Qx, double *Qy, double *Qz,
        int Nx, int Ny, int Nz
) {
    int err;

    err = MemoryCopy(d_Ex, Ex, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    CHECK_ERROR("1copyFieldsToGPU", err);

    err = MemoryCopy(d_Ey, Ey, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    CHECK_ERROR("2copyFieldsToGPU", err);

    err = MemoryCopy(d_Ez, Ez, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    CHECK_ERROR("3copyFieldsToGPU", err);

    err = MemoryCopy(d_Hx, Hx, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    CHECK_ERROR("4copyFieldsToGPU", err);

    err = MemoryCopy(d_Hy, Hy, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    CHECK_ERROR("5copyFieldsToGPU", err);

    err = MemoryCopy(d_Hz, Hz, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    CHECK_ERROR("6copyFieldsToGPU", err);

    err = MemoryCopy(d_Jx, Jx, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    CHECK_ERROR("7copyFieldsToGPU", err);

    err = MemoryCopy(d_Jy, Jy, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    CHECK_ERROR("8copyFieldsToGPU", err);

    err = MemoryCopy(d_Jz, Jz, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    CHECK_ERROR("9copyFieldsToGPU", err);

    err = MemoryCopy(d_npJx, npJx, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    CHECK_ERROR("10copyFieldsToGPU", err);

    err = MemoryCopy(d_npJy, npJy, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    CHECK_ERROR("11copyFieldsToGPU", err);

    err = MemoryCopy(d_npJz, npJz, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    CHECK_ERROR("12copyFieldsToGPU", err);

    err = MemoryCopy(d_Qx, Qx, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    CHECK_ERROR("13copyFieldsToGPU", err);

    err = MemoryCopy(d_Qy, Qy, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    CHECK_ERROR("14copyFieldsToGPU", err);

    err = MemoryCopy(d_Qz, Qz, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    CHECK_ERROR("15copyFieldsToGPU", err);
}

void InitGPUFields(
        double **d_Ex, double **d_Ey, double **d_Ez,
        double **d_Hx, double **d_Hy, double **d_Hz,
        double **d_Jx, double **d_Jy, double **d_Jz,
        double **d_npJx, double **d_npJy, double **d_npJz,
        double **d_Qx, double **d_Qy, double **d_Qz,
        double *Ex, double *Ey, double *Ez,
        double *Hx, double *Hy, double *Hz,
        double *Jx, double *Jy, double *Jz,
        double *npJx, double *npJy, double *npJz,
        double *Qx, double *Qy, double *Qz,
        int Nx, int Ny, int Nz
) {
    cudaMalloc3D(d_Ex, d_Ey, d_Ez, Nx, Ny, Nz);
    cudaMalloc3D(d_Hx, d_Hy, d_Hz, Nx, Ny, Nz);
    cudaMalloc3D(d_Jx, d_Jy, d_Jz, Nx, Ny, Nz);
    cudaMalloc3D(d_npJx, d_npJy, d_npJz, Nx, Ny, Nz);
    cudaMalloc3D(d_Qx, d_Qy, d_Qz, Nx, Ny, Nz);

    copyFieldsToGPU(
            *d_Ex, *d_Ey, *d_Ez,
            *d_Hx, *d_Hy, *d_Hz,
            *d_Jx, *d_Jy, *d_Jz,
            *d_npJx, *d_npJy, *d_npJz,
            *d_Qx, *d_Qy, *d_Qz,
            Ex, Ey, Ez,
            Hx, Hy, Hz,
            Jx, Jy, Jz,
            npJx, npJy, npJz,
            Qx, Qy, Qz,
            Nx, Ny, Nz
    );
}


