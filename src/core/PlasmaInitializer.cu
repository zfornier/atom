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

    InitCurrents();

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
    std::vector <Particle> ion_vp1, el_vp1, beam_vp1;
    double Lx = p->lx, Ly = p->ly, Lz = p->lz;
    int Nx = p->nx, Ny = p->ny, Nz = p->nz;

    initMeshArrays();

    getUniformMaxwellianParticles(ion_vp1, el_vp1, beam_vp1, p->tempX, p->tempY, p->tempZ, p->beamVelDisp, p->beamImp, p->beamPlasmaDensityRat, p->plsmDensity, p->lp, p->meh, Lx, Ly, Lz, Nx, Ny, Nz);

    addAllParticleListsToCells(ion_vp1, el_vp1, beam_vp1);

    AssignArraysToCells();
}

void PlasmaInitializer::Initialize() {
    InitializeCPU();

    copyCellsWithParticlesToGPU();

    InitializeGPU();
}

void PlasmaInitializer::InitGPUParticles() {
    int size, err;
    GPUCell *d_c, *h_ctrl;
    GPUCell *n;

    size = (int)(*p->AllCells).size();
    size_t m_free, m_total;

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

        err = cudaMemGetInfo(&m_free, &m_total);
        CHECK_ERROR("CUDA MEM GET INFO", err);

#ifdef COPY_CELL_PRINTS
        double mfree, mtot;
        mtot  = m_total;
        mfree = m_free;
        printf("cell %10d Device cell array allocated error %d %s memory: free %10.2f total %10.2f\n",i,err,getErrorString(err),mfree/1024/1024/1024,mtot/1024/1024/1024);

        dbgPrintGPUParticleAttribute(d_c,50,1," CO2DEV " );
        std::cout << "COPY----------------------------------" << std::endl;
#endif

        p->h_CellArray[i] = d_c;
        err = MemoryCopy(h_ctrl, d_c, sizeof(Cell), DEVICE_TO_HOST);
        CHECK_ERROR("MEM COPY", err);

#ifdef InitGPUParticles_PRINTS
        dbgPrintGPUParticleAttribute(d_c,50,1," CPY " );

        printf("i %d l %d k n %d %d %e src %e num %d\n",h_ctrl->i,h_ctrl->l,h_ctrl->k,i, c.ParticleArrayRead(0,7),c.number_of_particles);
        printf("GPU cell %d ended ******************************************************\n",i);
#endif
    }

    err = MemoryCopy(p->d_CellArray, p->h_CellArray, size * sizeof(Cell * ), HOST_TO_DEVICE);
    CHECK_ERROR("MEM COPY", err);

}

void PlasmaInitializer::Alloc() {
    p->AllCells = new std::vector<GPUCell>;
    int Nx = p->nx, Ny = p->ny, Nz = p->nz;

    p->Ex = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    p->Ey = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    p->Ez = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    p->Hx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    p->Hy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    p->Hz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    p->Jx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    p->Jy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    p->Jz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    p->Rho = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];

    p->npJx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    p->npJy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    p->npJz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];

    p->Qx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    p->Qy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    p->Qz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];

#ifdef DEBUG_PLASMA
    p->dbgEx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    p->dbgEy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    p->dbgEz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];

    p->dbgHx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    p->dbgHy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    p->dbgHz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    p->dbgJx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    p->dbgJy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    p->dbgJz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];

    p->dbg_Qx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    p->dbg_Qy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    p->dbg_Qz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
#endif
}

void PlasmaInitializer::InitFields() {
    int Nx = p->nx, Ny = p->ny, Nz = p->nz;

    for (int i = 0; i < (Nx + 2) * (Ny + 2) * (Nz + 2); i++) {
        p->Ex[i] = 0.0;
        p->Ey[i] = 0.0;
        p->Ez[i] = 0.0;
        p->Hx[i] = 0.0;
        p->Hy[i] = 0.0;
        p->Hz[i] = 0.0;

#ifdef DEBUG_PLASMA
        p->dbgEx[i] = 0.0;
        p->dbgEy[i] = 0.0;
        p->dbgEz[i] = 0.0;
        p->dbgHx[i] = 0.0;
        p->dbgHy[i] = 0.0;
        p->dbgHz[i] = 0.0;
#endif
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
#ifdef INIT_CELLS_DEBUG_PRINT
                printf("%5d %5d %5d size %d \n",i,l,k,(*p->AllCells).size());
#endif
            }
        }
    }
}

void PlasmaInitializer::InitCurrents() {
    int Nx = p->nx, Ny = p->ny, Nz = p->nz;

    for (int i = 0; i < (Nx + 2) * (Ny + 2) * (Nz + 2); i++) {
        p->Jx[i] = 0.0;
        p->Jy[i] = 0.0;
        p->Jz[i] = 0.0;
        p->Rho[i] = 0.0;

        p->dbgJx[i] = 0.0;
        p->dbgJy[i] = 0.0;
        p->dbgJz[i] = 0.0;
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

int PlasmaInitializer::copyCellsWithParticlesToGPU() {
    int Nx = p->nx, Ny = p->ny, Nz = p->nz;

    Cell c000 = (*p->AllCells)[0];

    int size = (Nx + 2) * (Ny + 2) * (Nz + 2);

    p->cp = (GPUCell **) malloc(size * sizeof(GPUCell *));

    for (int i = 0; i < size; i++) {
        GPUCell c, *d_c;
        d_c = c.allocateCopyCellFromDevice();

        p->cp[i] = d_c;
    }

    return 0;
}
