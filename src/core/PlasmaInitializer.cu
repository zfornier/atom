//
// Created by egor on 18.03.19.
//

#include "../../include/PlasmaInitializer.h"

PlasmaInitializer::PlasmaInitializer(PlasmaConfig * p) {
    this->p = p;
}

int PlasmaInitializer::InitializeGPU() {
    int err = getLastError();
    int Nx = p->nx, Ny = p->ny, Nz = p->nz;

    if (err != cudaSuccess) {
        printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
    }

    InitGPUParticles();

    err = getLastError();
    if (err != cudaSuccess) {
        printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
    }

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

    err = getLastError();
    if (err != cudaSuccess) {
        printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
    }

    setPrintfLimit();

    err = cudaSetDevice(0);

    printf("InitializeGPU error %d \n", err);

    return 0;
}

int PlasmaInitializer::initMeshArrays() {
    initControlPointFile();

    Alloc();

    Cell c000;

    InitCells();
    c000 = (*p->AllCells)[0];

    InitFields();
    c000 = (*p->AllCells)[0]; // TODO: why twice?

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

    initMeshArrays();

    getUniformMaxwellianParticles(ion_vp1, el_vp1, beam_vp1, p->tempX, p->tempY, p->tempZ, p->beamVelDisp, p->beamImp, p->beamPlasmaDensityRat, p->plsmDensity, Lx, Ly, Lz);

    addAllParticleListsToCells(ion_vp1, el_vp1, beam_vp1);

    AssignArraysToCells();
}

void PlasmaInitializer::Initialize() {
    int err;

    InitializeCPU();

    copyCellsWithParticlesToGPU();

    err = getLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err)); }

    InitializeGPU();

    err = getLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err)); }
}

void PlasmaInitializer::InitGPUParticles() {
    int size;
    GPUCell *d_c, *h_ctrl;
    GPUCell *n;
    int Nx = p->nx, Ny = p->ny, Nz = p->nz;

    dim3 dimGrid(Nx + 2, Ny + 2, Nz + 2), dimBlockOne(1, 1, 1);

    int err = getLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err)); }

    err = getLastError();
    if (err != cudaSuccess) {
        printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
        exit(0);
    }
    readControlFile(START_STEP_NUMBER);
    err = getLastError();
    if (err != cudaSuccess) {
        printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
        exit(0);
    }

    err = getLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err)); }

    size = (int)(*p->AllCells).size();
    err = getLastError();
    if (err != cudaSuccess) {
        printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
        exit(0);
    }
    size_t m_free, m_total;

    h_ctrl = new GPUCell;
    n = new GPUCell;

    err = getLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err)); }
    err = getLastError();
    if (err != cudaSuccess) {
        printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
        exit(0);
    }

    p->h_CellArray = (GPUCell **) malloc(size * sizeof(Cell * ));
    err = cudaMalloc((void **) &p->d_CellArray, size * sizeof(Cell * ));

    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err)); }

    err = getLastError();
    if (err != cudaSuccess) {
        printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
        exit(0);
    }
    printf("%s : size = %d\n", __FILE__, size);
    for (int i = 0; i < size; i++) {
        GPUCell c;
        c = (*p->AllCells)[i];

        /////////////////////////////////////////
        *n = c;
        err = getLastError();
        if (err != cudaSuccess) {
            printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
            exit(0);
        }
#ifdef ATTRIBUTES_CHECK
        c.SetControlSystem(p->jmp, p->d_ctrlParticles);
#endif
        d_c = c.copyCellToDevice();
        err = getLastError();
        if (err != cudaSuccess) {
            printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
            exit(0);
        }
        err = getLastError();
        if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err)); }

        cudaMemGetInfo(&m_free, &m_total);

        err = getLastError();
        if (err != cudaSuccess) {
            printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
            exit(0);
        }
#ifdef COPY_CELL_PRINTS
        double mfree, mtot;
        mtot  = m_total;
        mfree = m_free;
        printf("cell %10d Device cell array allocated error %d %s memory: free %10.2f total %10.2f\n",i,err,getErrorString(err),mfree/1024/1024/1024,mtot/1024/1024/1024);
        puts("");

        dbgPrintGPUParticleAttribute(d_c,50,1," CO2DEV " );
        puts("COPY----------------------------------");
#endif

#ifdef PARTICLE_PRINTS
        if(t < 1.0) {
            t = c.compareToCell(*h_copy);
        }
#endif
        err = getLastError();
        if (err != cudaSuccess) {
            printf("%s:%d - error %d %s cell %d \n", __FILE__, __LINE__, err, getErrorString(err), i);
            exit(0);
        }

        p->h_CellArray[i] = d_c;
        err = MemoryCopy(h_ctrl, d_c, sizeof(Cell), DEVICE_TO_HOST);

        //  err = getLastError();
        if (err != cudaSuccess) {
            printf("%s:%d - error %d %s cell %d\n", __FILE__, __LINE__, err, getErrorString(err), i);
            exit(0);
        }
#ifdef InitGPUParticles_PRINTS
        dbgPrintGPUParticleAttribute(d_c,50,1," CPY " );

        cudaPrintfInit();

        testKernel<<<1,1>>>(h_ctrl->d_ctrlParticles,h_ctrl->jmp);
        cudaPrintfDisplay(stdout, true);
        cudaPrintfEnd();

        printf("i %d l %d k n %d %d %e src %e num %d\n",h_ctrl->i,h_ctrl->l,h_ctrl->k,i, c.ParticleArrayRead(0,7),c.number_of_particles);
        printf("GPU cell %d ended ******************************************************\n",i);
#endif
        err = getLastError();
        if (err != cudaSuccess) {
            printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
            exit(0);
        }
    }

    err = getLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err)); }

    //int err;
    err = MemoryCopy(p->d_CellArray, p->h_CellArray, size * sizeof(Cell * ), HOST_TO_DEVICE);
    if (err != cudaSuccess) {
        printf("bGPU_WriteControlSystem err %d %s \n", err, getErrorString(err));
        exit(0);
    }

    err = getLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err)); }

#ifdef ATTRIBUTES_CHECK
    GPU_WriteControlSystem<<<dimGrid, dimBlockOne,16000>>>(d_CellArray);
#endif
    size = 0;

    err = getLastError();
    if (err != cudaSuccess) {
        printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
    }
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
    p-> dbgEy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
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

        p->dbgEx[i] = 0.0;
        p->dbgEy[i] = 0.0;
        p->dbgEz[i] = 0.0;
        p->dbgHx[i] = 0.0;
        p->dbgHy[i] = 0.0;
        p->dbgHz[i] = 0.0;
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
    cudaError_t err;

    int size = (Nx + 2) * (Ny + 2) * (Nz + 2);

    p->cp = (GPUCell **) malloc(size * sizeof(GPUCell *));

    if ((err = cudaGetLastError()) != cudaSuccess) {
        printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err));
    }

    for (int i = 0; i < size; i++) {
        GPUCell c, *d_c;
        d_c = c.allocateCopyCellFromDevice();
        if ((err = cudaGetLastError()) != cudaSuccess) {
            printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err));
        }

        p->cp[i] = d_c;
    }

    if ((err = cudaGetLastError()) != cudaSuccess) {
        printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err));
    }

    return 0;
}

int PlasmaInitializer::readControlFile(int nt) {

#ifndef ATTRIBUTES_CHECK
    return 0;
#else
    FILE *f;
    char fname[100];
    static int first = 1;
    int size;

    sprintf(fname,"ctrl%05d",nt);

    if((f = fopen(fname,"rb")) == NULL) {
        puts("no ini-file");
        exit(0);
    }

    fread(&size,sizeof(int),1,f);
    fread(&p->ami,sizeof(double),1,f);
    fread(&p->amf,sizeof(double),1,f);
    fread(&p->amb,sizeof(double),1,f);
    fread(&size,sizeof(int),1,f);

    fread(&size,sizeof(int),1,f);

    if(first == 1) {
        first = 0;
        p->ctrlParticles = (double *)malloc(size);
#ifdef ATTRIBUTES_CHECK
        memset(p->ctrlParticles,0,size);
        cudaMalloc((void **)&p->d_ctrlParticles,size);
        cudaMemset(p->d_ctrlParticles,0,size);
        p->size_ctrlParticles = size;
#endif
    }
    fread(p->ctrlParticles,1,size,f);


    p->jmp = size / sizeof(double) / PARTICLE_ATTRIBUTES / 3;

    return 0;
#endif
}