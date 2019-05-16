//
// Created by egor on 12.05.19.
//

#ifndef ATOM_NETCDFCONFIG_H
#define ATOM_NETCDFCONFIG_H

typedef struct {
    int nx, ny, nz, ionTotal, electronsTotal, beamTotal;
    double *Ex, *Ey, *Ez, *Hx, *Hy, *Hz, *Qx, *Qy, *Qz, *Jx, *Jy, *Jz;
    double massIons, massElectrons, massBeam;
    double chargeIons, chargeElectrons, chargeBeam;
    double *ionsX, *ionsY, *ionsZ, *ionsPx, *ionsPy, *ionsPz;
    double *electronsX, *electronsY, *electronsZ, *electronsPx, *electronsPy, *electronsPz;
    double *beamX, *beamY, *beamZ, *beamPx, *beamPy, *beamPz;
} NetCdfData;

#endif //ATOM_NETCDFCONFIG_H
