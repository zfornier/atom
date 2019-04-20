#include <stdlib.h>
#include <getopt.h>
#include <cstring>

#include "../include/mpi_shortcut.h"
#include "../include/Plasma.h"
#include "../include/PlasmaTypes.h"
#include "../include/ConfigParser/Properties.h"

//TODO: gpu cell in the global array at copy from there appears to be not initialized
using namespace std;

void printHelp();
PlasmaConfig readConfig(std::ifstream &is);
bool isFileExist(const char *);

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

        if(!isFileExist(conf.checkFile)) {
            std::cerr << "Check file: " << conf.checkFile << " does not exist" << std::endl;
            return -1;
        }

        InitMPI(argc, argv);

        cout << "begin Particle size " <<  sizeof(Particle) << endl;

        try {
            plasma = new Plasma(&conf);

            plasma->Initialize();

            plasma->Compute();

            delete plasma;
        } catch (std::bad_alloc &e) {
            std::cerr << "Unable to allocate memory" << std::endl;
        } catch (std::exception &e) {
            std::cout << e.what() << std::endl;
        }

        delete conf.checkFile;

        CloseMPI();

    } else {
        cout << "Try -h or --help for usage information." << endl;
    }

    return 0;
}

void printHelp() {
    cout << "Usage: ./atom -i <configuration file>" << endl;
    cout << "Options:\n" << "-h --help\tdisplay usage information" << endl;
}

PlasmaConfig readConfig(std::ifstream &is) {
    Properties properties;
    PlasmaConfig conf = PlasmaConfig();

    try {
        properties.load(is);

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
        std::string file = properties.getProperty("checkFile");
        int l = (int)file.length() + 1;
        conf.checkFile = new char[l + 1];
        memcpy((void*)conf.checkFile, (void*)file.c_str(), (size_t)l + 1);
    }
    catch (std::exception &e) {
        std::cout << "Config parsing error: " << e.what() << std::endl;
        exit(-1);
    }

    return conf;
}

bool isFileExist(const char * name) {
    ifstream f(name);
    return f.good();
}