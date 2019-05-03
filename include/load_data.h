/*
 * load_data.h
 *
 *  Created on: Jun 9, 2018
 *      Author: snytav
 */

#ifndef LOAD_DATA_H_
#define LOAD_DATA_H_

#include <sstream>

#include <sys/sysinfo.h>

#include "particle.h"
#include "maxwell.h"
#include "NetCdf/read_file.h"
#include "archAPI.h"

void debugPrintParticleCharacteristicArray(double *p_ch, int np, int nt, std::string name, int sort);

void readParticleParamsOneSort(const char *, int * total_particles, double *qq_m, double *mm, int sort);

void readBinaryParticleArraysOneSort(
        const char * filename,
        double *dbg_x, double *dbg_y, double *dbg_z, double *dbg_px, double *dbg_py, double *dbg_pz,
        int total_particles,
        int nt,
        int sort
);

namespace patch {
    template<typename T>
    std::string to_string(const T &n) {
        std::ostringstream stm;
        stm << n;
        return stm.str();
    }
}

#endif /* LOAD_DATA_H_ */
