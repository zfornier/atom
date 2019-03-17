
int InitializeGPU() {
    int err = getLastError();
    if (err != cudaSuccess) {
        printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
    }

    InitGPUParticles();

    err = getLastError();
    if (err != cudaSuccess) {
        printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
    }

    InitGPUFields(
            &d_Ex, &d_Ey, &d_Ez,
            &d_Hx, &d_Hy, &d_Hz,
            &d_Jx, &d_Jy, &d_Jz,
            &d_npJx, &d_npJy, &d_npJz,
            &d_Qx, &d_Qy, &d_Qz,
            Ex, Ey, Ez,
            Hx, Hy, Hz,
            Jx, Jy, Jz,
            npJx, npJy, npJz,
            Qx, Qy, Qz,
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

int initMeshArrays() {
    initControlPointFile();

    Alloc();

    Cell c000;

    InitCells();
    c000 = (*AllCells)[0];

    InitFields();
    c000 = (*AllCells)[0];
    InitCurrents();

    return 0;
}

void AssignArraysToCells() {
    for (int n = 0; n < (*AllCells).size(); n++) {
        Cell c = (*AllCells)[n];
        c.readFieldsFromArrays(Ex, Ey, Ez, Hx, Hy, Hz);
    }
}

virtual void InitializeCPU(double tex0, double tey0, double tez0, double Tb, double rimp, double rbd) {
    std::vector <Particle> ion_vp, el_vp, beam_vp;
    std::vector <Particle> ion_vp1, el_vp1, beam_vp1;

    initMeshArrays();

    getUniformMaxwellianParticles(ion_vp1, el_vp1, beam_vp1, tex0, tey0, tez0, Tb, rimp, rbd, ni, Lx, Ly, Lz);

    addAllParticleListsToCells(ion_vp1, el_vp1, beam_vp1);

    AssignArraysToCells();
}

void Initialize(double tex0, double tey0, double tez0, double Tb, double rimp, double rbd) {
    int err = getLastError();

    InitializeCPU(tex0, tey0, tez0, Tb, rimp, rbd);

    copyCellsWithParticlesToGPU();

    err = getLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err)); }

    InitializeGPU();

    err = getLastError();
    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err)); }
}

void InitGPUParticles() {
    int size;
    GPUCell *d_c, *h_ctrl;
    GPUCell *n;

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

    size = (*AllCells).size();
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

    h_CellArray = (GPUCell **) malloc(size * sizeof(Cell * ));
    err = cudaMalloc((void **) &d_CellArray, size * sizeof(Cell * ));

    if (err != cudaSuccess) { printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err)); }

    err = getLastError();
    if (err != cudaSuccess) {
        printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
        exit(0);
    }
    printf("%s : size = %d\n", __FILE__, size);
    for (int i = 0; i < size; i++) {
        GPUCell c;
        c = (*AllCells)[i];

        h_controlParticleNumberArray[i] = c.number_of_particles;
        /////////////////////////////////////////
        *n = c;
        err = getLastError();
        if (err != cudaSuccess) {
            printf("%s:%d - error %d %s\n", __FILE__, __LINE__, err, getErrorString(err));
            exit(0);
        }
#ifdef ATTRIBUTES_CHECK
        c.SetControlSystem(jmp,d_ctrlParticles);
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
        double mfree,mtot;
        mtot  = m_total;
        mfree = m_free;
        printf("cell %10d Device cell array allocated error %d %s memory: free %10.2f total %10.2f\n",i,err,getErrorString(err),
                                                                mfree/1024/1024/1024,mtot/1024/1024/1024);
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

        h_CellArray[i] = d_c;
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
    err = MemoryCopy(d_CellArray, h_CellArray, size * sizeof(Cell * ), HOST_TO_DEVICE);
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

virtual void Alloc() {
    AllCells = new std::vector<GPUCell>;

    Ex = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    Ey = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    Ez = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    Hx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    Hy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    Hz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    Jx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    Jy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    Jz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    Rho = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];

    npJx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    npJy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    npJz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];

    npEx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    npEy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    npEz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];

    Qx = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    Qy = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];
    Qz = new double[(Nx + 2) * (Ny + 2) * (Nz + 2)];

#ifdef DEBUG_PLASMA
    dbgEx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    dbgEy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    dbgEz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    dbgEx0  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    dbgEy0  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    dbgEz0  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];

    dbgHx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    dbgHy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    dbgHz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    dbgJx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    dbgJy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    dbgJz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];

    dbg_Qx  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    dbg_Qy  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
    dbg_Qz  = new double[(Nx + 2)*(Ny + 2)*(Nz + 2)];
#endif
}

virtual void InitFields() {
    for (int i = 0; i < (Nx + 2) * (Ny + 2) * (Nz + 2); i++) {
        Ex[i] = 0.0;
        Ey[i] = 0.0;
        Ez[i] = 0.0;
        Hx[i] = 0.0;
        Hy[i] = 0.0;
        Hz[i] = 0.0;

        dbgEx[i] = 0.0;
        dbgEy[i] = 0.0;
        dbgEz[i] = 0.0;
        dbgHx[i] = 0.0;
        dbgHy[i] = 0.0;
        dbgHz[i] = 0.0;
    }
}

virtual void InitCells() {
    for (int i = 0; i < Nx + 2; i++) {
        for (int l = 0; l < Ny + 2; l++) {
            for (int k = 0; k < Nz + 2; k++) {
                GPUCell *c = new GPUCell(i, l, k, Lx, Ly, Lz, Nx, Ny, Nz, tau);
                c->Init();
                (*AllCells).push_back(*c);
#ifdef INIT_CELLS_DEBUG_PRINT
                printf("%5d %5d %5d size %d \n",i,l,k,(*AllCells).size());
#endif
            }
        }
    }
}

virtual void InitCurrents() {
    for (int i = 0; i < (Nx + 2) * (Ny + 2) * (Nz + 2); i++) {
        Jx[i] = 0.0;
        Jy[i] = 0.0;
        Jz[i] = 0.0;
        Rho[i] = 0.0;

        dbgJx[i] = 0.0;
        dbgJy[i] = 0.0;
        dbgJz[i] = 0.0;
    }
}

int addParticleListToCells(std::vector <Particle> &vp) {
    Cell c0 = (*AllCells)[0];
    int n;

    for (int i = 0; i < vp.size(); i++) {
        Particle p = vp[i];

        double3 d;
        d.x = p.x;
        d.y = p.y;
        d.z = p.z;

        n = c0.getPointCell(d);

        Cell &c = (*AllCells)[n];

        if (c.Insert(p) == true) {
#ifdef PARTICLE_PRINTS1000
            if((i+1)%1000 == 0) {
                printf("particle %d (%e,%e,%e) is number %d in cell (%d,%d,%d)\n", i+1, x,y,z,c.number_of_particles,c.i,c.l,c.k);
            }
#endif
        }
    }   // END total_particles LOOP

    return 0;
}

int addAllParticleListsToCells(std::vector <Particle> &ion_vp, std::vector <Particle> &el_vp, std::vector <Particle> &beam_vp) {
    addParticleListToCells(ion_vp);
    addParticleListToCells(el_vp);
    addParticleListToCells(beam_vp);

    return 0;
}