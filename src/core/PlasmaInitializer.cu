//
// Created by egor on 18.03.19.
//

#include "../../include/PlasmaInitializer.h"

PlasmaInitializer::PlasmaInitializer(PlasmaConfig * p) {
    this->p = p;
}

int PlasmaInitializer::InitializeGPU() {
    int err;
    int Nx = p->nx, Ny = p->ny, Nz = p->nz;

    InitGPUParticles();

    InitGPUFields(
            &p->d_Ex, &p->d_Ey, &p->d_Ez,
            &p->d_Hx, &p->d_Hy, &p->d_Hz,
            &p->d_Jx, &p->d_Jy, &p->d_Jz,
            &p->d_npJx, &p->d_npJy, &p->d_npJz,
            &p->d_Qx, &p->d_Qy, &p->d_Qz,
            p->Ex, p->Ey, p->Ez,
            p->Hx, p->Hy, p->Hz,
            p->Jx, p->Jy, p->Jz,
            p->npJx, p->npJy, p->npJz,
            p->Qx, p->Qy, p->Qz,
            Nx, Ny, Nz);

    setPrintfLimit();

    err = cudaSetDevice(0);
    CHECK_ERROR("DEVICE SET", err);

    return 0;
}

int PlasmaInitializer::initMeshArrays() {
    initControlPointFile();

    Alloc();

    InitCells();

    InitFields();

    return 0;
}

void PlasmaInitializer::AssignArraysToCells() {
    for (int n = 0; n < (*p->AllCells).size(); n++) {
        Cell c = (*p->AllCells)[n];
        c.readFieldsFromArrays(p->Ex, p->Ey, p->Ez, p->Hx, p->Hy, p->Hz);
    }
}

void PlasmaInitializer::InitializeCPU() {
    std::vector <Particle> ion_vp, el_vp, beam_vp;

    initMeshArrays();

    ParticlesConfig pC;
    pC.nx = p->nx;
    pC.ny = p->ny;
    pC.nz = p->nz;
    pC.lp = p->lp;
    pC.lx = p->lx;
    pC.ly = p->ly;
    pC.lz = p->lz;
    pC.beam_lx = p->lx;
    pC.beam_ly = p->ly;
    pC.beam_lz = p->lz;
    pC.ions = &(p->ions);
    pC.electrons = &(p->electrons);
    pC.beam = &(p->beam);
    pC.tempX = p->tempX;
    pC.tempY = p->tempY;
    pC.tempZ = p->tempZ;
    pC.meh = p->meh;
    pC.beamVelDisp = p->beamVelDisp;
    pC.beamImp= p->beamImp;
    pC.beamPlasmaDensityRat = p->beamPlasmaDensityRat;
    pC.plsmDensity= p->plsmDensity;
    pC.beamPlasma = p->beamPlasma;

    getUniformMaxwellianParticles(ion_vp, el_vp, beam_vp, &pC);

    addAllParticleListsToCells(ion_vp, el_vp, beam_vp);

    AssignArraysToCells();
}

void PlasmaInitializer::InitializeCPU(NetCdfData * data) {
    std::vector <Particle> ion_vp, el_vp, beam_vp;

    initMeshArrays();

    p->Ex = data->Ex;
    p->Ey = data->Ey;
    p->Ez = data->Ez;

    p->Hx = data->Hx;
    p->Hy = data->Hy;
    p->Hz = data->Hz;

    p->Qx = data->Qx;
    p->Qy = data->Qy;
    p->Qz = data->Qz;

    p->Jx = data->Jx;
    p->Jy = data->Jy;
    p->Jz = data->Jz;

    ParticlesConfig pC;

    pC.ions = &(p->ions);
    pC.electrons = &(p->electrons);
    pC.beam = &(p->beam);

    pC.ions->m = &data->massIons;
    pC.ions->q_m = data->chargeIons;
    pC.ions->total = data->ionTotal;
    pC.ions->x = data->ionsX;
    pC.ions->y = data->ionsY;
    pC.ions->z = data->ionsZ;
    pC.ions->px = data->ionsPx;
    pC.ions->py = data->ionsPy;
    pC.ions->pz = data->ionsPz;

    pC.electrons->m = &data->massElectrons;
    pC.electrons->q_m = data->chargeElectrons;
    pC.electrons->total = data->electronsTotal;
    pC.electrons->x = data->electronsX;
    pC.electrons->y = data->electronsY;
    pC.electrons->z = data->electronsZ;
    pC.electrons->px = data->electronsPx;
    pC.electrons->py = data->electronsPy;
    pC.electrons->pz = data->electronsPz;

    pC.beam->m = &data->massBeam;
    pC.beam->q_m = data->chargeBeam;
    pC.beam->total = data->beamTotal;
    pC.beam->x = data->beamX;
    pC.beam->y = data->beamY;
    pC.beam->z = data->beamZ;
    pC.beam->px = data->beamPx;
    pC.beam->py = data->beamPy;
    pC.beam->pz = data->beamPz;

    convertParticleArraysToSTLvector(pC.beam, BEAM_ELECTRON, beam_vp);
    convertParticleArraysToSTLvector(pC.ions, ION, ion_vp);
    convertParticleArraysToSTLvector(pC.electrons, PLASMA_ELECTRON, el_vp);

    addAllParticleListsToCells(ion_vp, el_vp, beam_vp);

    AssignArraysToCells();
}

void PlasmaInitializer::Initialize() {
    InitializeCPU();

    allocMemoryForCopyCells();

    InitializeGPU();
}

void PlasmaInitializer::Initialize(NetCdfData * data) {
    InitializeCPU(data);

    allocMemoryForCopyCells();

    InitializeGPU();
}

void PlasmaInitializer::InitGPUParticles() {
    int size, err;
    GPUCell *d_c, *h_ctrl;
    GPUCell *n;

    size = (int)(*p->AllCells).size();
    h_ctrl = new GPUCell;
    n = new GPUCell;


    p->h_CellArray = (GPUCell **) malloc(size * sizeof(Cell * ));
    err = cudaMalloc((void **) &p->d_CellArray, size * sizeof(Cell * ));
    CHECK_ERROR("CUDA MALLOC", err);

    printf("%s : size = %d\n", __FILE__, size);

    for (int i = 0; i < size; i++) {
        GPUCell c;
        c = (*p->AllCells)[i];

        *n = c;

        d_c = c.copyCellToDevice();

        p->h_CellArray[i] = d_c;
        err = MemoryCopy(h_ctrl, d_c, sizeof(Cell), DEVICE_TO_HOST);
        CHECK_ERROR("MEM COPY", err);
    }

    err = MemoryCopy(p->d_CellArray, p->h_CellArray, size * sizeof(Cell * ), HOST_TO_DEVICE);
    CHECK_ERROR("MEM COPY", err);

}

void PlasmaInitializer::Alloc() {
    p->AllCells = new std::vector<GPUCell>;
    int Nx = p->nx, Ny = p->ny, Nz = p->nz;

    if (p->computeFromFile == NULL) {
        p->Ex = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
        p->Ey = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
        p->Ez = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];

        p->Hx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
        p->Hy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
        p->Hz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];

        p->Jx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
        p->Jy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
        p->Jz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];

        p->Qx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
        p->Qy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
        p->Qz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    }

    p->Rho = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];

    p->npJx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    p->npJy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    p->npJz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];

    if (p->checkFile != NULL) {
        p->dbgEx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
        p->dbgEy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
        p->dbgEz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];

        p->dbgHx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
        p->dbgHy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
        p->dbgHz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];

        p->dbgJx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
        p->dbgJy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
        p->dbgJz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];

        p->dbg_Qx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
        p->dbg_Qy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
        p->dbg_Qz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    }
}

void PlasmaInitializer::InitFields() {
    int Nx = p->nx, Ny = p->ny, Nz = p->nz;

    for (int i = 0; i < (Nx + 2) * (Ny + 2) * (Nz + 2); i++) {
        if (p->computeFromFile == NULL) {
            p->Ex[i] = 0.0;
            p->Ey[i] = 0.0;
            p->Ez[i] = 0.0;
            p->Hx[i] = 0.0;
            p->Hy[i] = 0.0;
            p->Hz[i] = 0.0;
            p->Jx[i] = 0.0;
            p->Jy[i] = 0.0;
            p->Jz[i] = 0.0;
        }

        p->Rho[i] = 0.0;

        if (p->checkFile != NULL) {
            p->dbgEx[i] = 0.0;
            p->dbgEy[i] = 0.0;
            p->dbgEz[i] = 0.0;
            p->dbgHx[i] = 0.0;
            p->dbgHy[i] = 0.0;
            p->dbgHz[i] = 0.0;
            p->dbgJx[i] = 0.0;
            p->dbgJy[i] = 0.0;
            p->dbgJz[i] = 0.0;
        }
    }
}

void PlasmaInitializer::InitCells() {
    int Nx = p->nx, Ny = p->ny, Nz = p->nz;
    double Lx = p->lx, Ly = p->ly, Lz = p->lz;

    for (int i = 0; i < Nx + 2; i++) {
        for (int l = 0; l < Ny + 2; l++) {
            for (int k = 0; k < Nz + 2; k++) {
                GPUCell *c = new GPUCell(i, l, k, Lx, Ly, Lz, Nx, Ny, Nz, p->tau);
                c->Init();
                (*p->AllCells).push_back(*c);
            }
        }
    }
}

int PlasmaInitializer::addParticleListToCells(std::vector <Particle> &vp) {
    Cell c0 = (*p->AllCells)[0];
    int n;

    for (int i = 0; i < vp.size(); i++) {
        Particle particle = vp[i];

        double3 d;
        d.x = particle.x;
        d.y = particle.y;
        d.z = particle.z;

        n = c0.getPointCell(d);

        Cell & c = (*p->AllCells)[n];

        if (c.Insert(particle)) {
#ifdef PARTICLE_PRINTS1000
            if((i + 1) % 1000 == 0) {
                printf("particle %d (%e,%e,%e) is number %d in cell (%d,%d,%d)\n", i+1, x,y,z,c.number_of_particles,c.i,c.l,c.k);
            }
#endif
        }
    }   // END total_particles LOOP

    return 0;
}

int PlasmaInitializer::addAllParticleListsToCells(std::vector <Particle> &ion_vp, std::vector <Particle> &el_vp, std::vector <Particle> &beam_vp) {
    addParticleListToCells(ion_vp);
    addParticleListToCells(el_vp);
    addParticleListToCells(beam_vp);

    return 0;
}

int PlasmaInitializer::initControlPointFile() {
    p->f_prec_report = fopen("control_points.dat", "wt");
    fclose(p->f_prec_report);

    return 0;
}

int PlasmaInitializer::allocMemoryForCopyCells() {
    int Nx = p->nx, Ny = p->ny, Nz = p->nz;

    int size = (Nx + 2) * (Ny + 2) * (Nz + 2);

    p->cp = (GPUCell **) malloc(size * sizeof(GPUCell *));

    for (int i = 0; i < size; i++) {
        GPUCell c, *d_c;
        d_c = c.allocateCopyCellFromDevice();

        p->cp[i] = d_c;
    }

    return 0;
}
