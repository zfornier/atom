//
// Created by egor on 18.03.19.
//

#ifndef ATOM_PLASMAINITIALIZATOR_H
#define ATOM_PLASMAINITIALIZATOR_H

#include "PlasmaTypes.h"
#include "maxwell.h"
#include "service_functions.h"

class PlasmaInitializer {
private:
    PlasmaConfig * p;
public:
    PlasmaInitializer(PlasmaConfig * plasma);

    void Initialize();

    void AssignArraysToCells();

private:

    int InitializeGPU();

    int initMeshArrays();

    virtual void InitializeCPU();

    void InitGPUParticles();

    virtual void Alloc();

    virtual void InitFields();

    virtual void InitCells();

    virtual void InitCurrents();

    int addParticleListToCells(std::vector <Particle> &vp);

    int addAllParticleListsToCells(std::vector <Particle> &ion_vp, std::vector <Particle> &el_vp, std::vector <Particle> &beam_vp);

    int initControlPointFile();

    int copyCellsWithParticlesToGPU();

    int readControlFile(int);
};

#endif //ATOM_PLASMAINITIALIZATOR_H
