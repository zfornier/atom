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
#include "NetCdf/NetCdfData.h"
#include "archAPI.h"

void readParticleParamsOneSort(const char *, int *, double *, double *, int);

void readBinaryParticleArraysOneSort(const char *, double *, double *, double *, double *, double *, double *, int);

NetCdfData * getDataFromFile(const char *);

namespace patch {
    template<typename T>
    std::string to_string(const T &n) {
        std::ostringstream stm;
        stm << n;
        return stm.str();
    }
}

#endif /* LOAD_DATA_H_ */
