#include <stdlib.h>
#include <getopt.h>

#include "../include/mpi_shortcut.h"
#include "../include/Plasma.h"
#include "../include/PlasmaTypes.h"
#include "../include/ConfigParser/Properties.h"

//TODO: gpu cell in the global array at copy from there appears to be not initialized
using namespace std;

void printHelp();
PlasmaConfig readConfig(std::ifstream &is);

int main(int argc, char *argv[]) {
    Plasma *plasma;
    char *config = NULL;
    int c, option_index;

    const char* short_options = "i:h";
    const struct option long_options[] = {{"help", no_argument ,NULL, 'h'}, {0, 0, 0, 0}};

    while ((c = getopt_long(argc,argv,short_options, long_options, &option_index)) != -1) {
        switch (c) {
            case 'i':
                config = optarg;
                break;
            case 'h':
                printHelp();
                exit(0);
            default:
                break;
        }
    }

    if (config != NULL) {
        string line;
        ifstream myfile(config);
        PlasmaConfig conf;

        if (myfile.is_open()) {
            conf = readConfig(myfile);
        } else {
            cout << "Unable to open file: " << config << endl;
            return 0;
        }

        InitMPI(argc, argv);

        cout << "begin Particle size " <<  sizeof(Particle) << endl;

        // TODO: why are those magic constants here? ----------------------------------------------\_______/---
        plasma = new Plasma(conf.nx, conf.ny, conf.nz, conf.lx, conf.ly, conf.lz, conf.plsmDensity, 2000, 1.0, conf.tau);

        plasma->Initialize(conf.tempX, conf.tempY, conf.tempZ, conf.beamVelDisp, conf.beamImp, conf.beamPlasmaDensityRat);

        plasma->Compute(conf.st, conf.ts);

        CloseMPI();

        delete plasma;
    } else {
        printf("Try -h or --help for usage information.\n");
    }

    return 0;
}

void printHelp() {
    cout << "Usage: ./atom -i <configuration file>" << endl;
    cout << "Options:\n" << "-h --help\tdisplay usage information" << endl;
}

PlasmaConfig readConfig(std::ifstream &is) {
    Properties properties;
    properties.load(is);

    PlasmaConfig conf = PlasmaConfig();
    try {
        conf.tempX = properties.stringToDouble(properties.getProperty("tempX"));
        conf.tempY = properties.stringToDouble(properties.getProperty("tempY"));
        conf.tempZ = properties.stringToDouble(properties.getProperty("tempZ"));
        conf.beamImp = properties.stringToDouble(properties.getProperty("beamImp"));
        conf.beamVelDisp = properties.stringToDouble(properties.getProperty("beamVelDisp"));
        conf.beamPlasmaDensityRat = properties.stringToDouble(properties.getProperty("beamPlasmaDensityRat"));
        conf.plsmDensity = properties.stringToDouble(properties.getProperty("plsmDensity"));
        conf.externalMagnFieldX = properties.stringToDouble(properties.getProperty("externalMagnFieldX"));
        conf.lx = properties.stringToDouble(properties.getProperty("lx"));
        conf.ly = properties.stringToDouble(properties.getProperty("ly"));
        conf.lz = properties.stringToDouble(properties.getProperty("lz"));
        conf.px = properties.stringToDouble(properties.getProperty("px"));
        conf.py = properties.stringToDouble(properties.getProperty("py"));
        conf.bx = properties.stringToDouble(properties.getProperty("bx"));
        conf.by = properties.stringToDouble(properties.getProperty("by"));
        conf.bz = properties.stringToDouble(properties.getProperty("bz"));
        conf.lp = properties.stringToInt(properties.getProperty("lp"));
        conf.nx = properties.stringToInt(properties.getProperty("nx"));
        conf.ny = properties.stringToInt(properties.getProperty("ny"));
        conf.nz = properties.stringToInt(properties.getProperty("nz"));
        conf.tau = properties.stringToDouble(properties.getProperty("tau"));
        conf.beamPlasma = properties.stringToInt(properties.getProperty("beamPlasma"));
        conf.startFromFile = properties.stringToInt(properties.getProperty("startFromFile"));
        conf.phase = properties.stringToInt(properties.getProperty("phase"));
        conf.ts = properties.stringToInt(properties.getProperty("ts"));
        conf.ms = properties.stringToInt(properties.getProperty("ms"));
        conf.nsteps = properties.stringToInt(properties.getProperty("nsteps"));
        conf.saveStep = properties.stringToInt(properties.getProperty("saveStep"));
        conf.startSave = properties.stringToInt(properties.getProperty("startSave"));
        conf.st = properties.stringToInt(properties.getProperty("st"));
        conf.checkOnStep = properties.stringToInt(properties.getProperty("checkOnStep"));
        conf.checkFile = properties.getProperty("checkFile");
    }
    catch (std::invalid_argument e) {
        throw 1;  // todo: make normal exception
    }
    catch (std::out_of_range e) {
        throw 1;  // todo: make normal exception
    }

    return conf;
}