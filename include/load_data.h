/*
 * load_data.h
 *
 *  Created on: Jun 9, 2018
 *      Author: snytav
 */

#ifndef LOAD_DATA_H_
#define LOAD_DATA_H_


#include "particle.h"
#include <string>
#include <sstream>
#include <vector>

#include "maxwell.h"

#include <string>

#include "NetCdf/read_file.h"

void debugPrintParticleCharacteristicArray(double *p_ch, int np, int nt, std::string name, int sort);

int readBinaryParticleArraysOneSort(
        FILE *f,
        double **dbg_x,
        double **dbg_y,
        double **dbg_z,
        double **dbg_px,
        double **dbg_py,
        double **dbg_pz,
        double *qq_m,
        double *mm,
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
