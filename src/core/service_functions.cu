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
    cudaMalloc((void **) X, sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));
    cudaMalloc((void **) Y, sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));
    cudaMalloc((void **) Z, sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));
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

// TODO: It's better to define a macros and wrap  all MemoryCopy-s with it
// Moreover you dont need to write  numbers and any line... print __FILE__ and __LINE__ and optionaly
// You will print error code if build_type is debug (and not here but in ArchAPI.cu)

    err = MemoryCopy(d_Ex, Ex, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("1copyFieldsToGPU err %d %s \n", err, getErrorString(err));
        exit(0);
    }
    err = MemoryCopy(d_Ey, Ey, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("2copyFieldsToGPU err %d %s \n", err, getErrorString(err));
        exit(0);
    }

    err = MemoryCopy(d_Ez, Ez, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("3copyFieldsToGPU err %d %s \n", err, getErrorString(err));
        exit(0);
    }

    err = MemoryCopy(d_Hx, Hx, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("4copyFieldsToGPU err %d %s \n", err, getErrorString(err));
        exit(0);
    }

    err = MemoryCopy(d_Hy, Hy, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("5copyFieldsToGPU err %d %s \n", err, getErrorString(err));
        exit(0);
    }

    err = MemoryCopy(d_Hz, Hz, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("6copyFieldsToGPU err %d %s \n", err, getErrorString(err));
        exit(0);
    }

    err = MemoryCopy(d_Jx, Jx, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("7copyFieldsToGPU err %d %s \n", err, getErrorString(err));
        exit(0);
    }

    err = MemoryCopy(d_Jy, Jy, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("8copyFieldsToGPU err %d %s \n", err, getErrorString(err));
        exit(0);
    }

    err = MemoryCopy(d_Jz, Jz, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("9copyFieldsToGPU err %d %s \n", err, getErrorString(err));
        exit(0);
    }

    err = MemoryCopy(d_npJx, npJx, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("10copyFieldsToGPU err %d %s \n", err, getErrorString(err));
        exit(0);
    }

    err = MemoryCopy(d_npJy, npJy, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("11copyFieldsToGPU err %d %s \n", err, getErrorString(err));
        exit(0);
    }

    err = MemoryCopy(d_npJz, npJz, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("12copyFieldsToGPU err %d %s \n", err, getErrorString(err));
        exit(0);
    }

    err = MemoryCopy(d_Qx, Qx, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("13copyFieldsToGPU err %d %s \n", err, getErrorString(err));
        exit(0);
    }

    err = MemoryCopy(d_Qy, Qy, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("14copyFieldsToGPU err %d %s \n", err, getErrorString(err));
        exit(0);
    }

    err = MemoryCopy(d_Qz, Qz, sizeof(double) * (Nx + 2) * (Ny + 2) * (Nz + 2), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("15copyFieldsToGPU err %d %s \n", err, getErrorString(err));
        exit(0);
    }
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


