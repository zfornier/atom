//
// Created by egor on 18.03.19.
//

#ifndef ATOM_PLASMAINITIALIZATOR_H
#define ATOM_PLASMAINITIALIZATOR_H

#include "Plasma.h"

class PlasmaInitializer {
private:
    Plasma * plasma;
public:
    PlasmaInitializer(Plasma * plasma);

    int InitializeGPU();

    int initMeshArrays();

    void AssignArraysToCells();

    virtual void InitializeCPU(double tex0, double tey0, double tez0, double Tb, double rimp, double rbd);

    void Initialize(double tex0, double tey0, double tez0, double Tb, double rimp, double rbd);

    void InitGPUParticles();

    virtual void Alloc();

    virtual void InitFields();

    virtual void InitCells();

    virtual void InitCurrents();

    int addParticleListToCells(std::vector <Particle> &vp);

    int addAllParticleListsToCells(std::vector <Particle> &ion_vp, std::vector <Particle> &el_vp, std::vector <Particle> &beam_vp);
};

#endif //ATOM_PLASMAINITIALIZATOR_H
