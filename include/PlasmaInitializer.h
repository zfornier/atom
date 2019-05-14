//
// Created by egor on 18.03.19.
//

#ifndef ATOM_PLASMAINITIALIZATOR_H
#define ATOM_PLASMAINITIALIZATOR_H

#include "PlasmaTypes.h"
#include "maxwell.h"
#include "service_functions.h"
#include "archAPI.h"
#include "NetCdf/read_file.h"
#include "NetCdf/NetCdfData.h"

class PlasmaInitializer {
private:
    PlasmaConfig * p;
public:
    PlasmaInitializer(PlasmaConfig * plasma);

    void Initialize();

    void Initialize(NetCdfData *);

    void AssignArraysToCells();

private:

    int InitializeGPU();

    int initMeshArrays();

    virtual void InitializeCPU();

    virtual void InitializeCPU(NetCdfData *);

    void InitGPUParticles();

    virtual void Alloc();

    virtual void InitFields();

    virtual void InitCells();

    int addParticleListToCells(std::vector <Particle> &vp);

    int addAllParticleListsToCells(std::vector <Particle> &ion_vp, std::vector <Particle> &el_vp, std::vector <Particle> &beam_vp);

    int initControlPointFile();

    int allocMemoryForCopyCells();

};

#endif //ATOM_PLASMAINITIALIZATOR_H
