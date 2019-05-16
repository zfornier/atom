#include <cstdlib>
#include <cstring>
#include <getopt.h>

#include "../include/mpi_shortcut.h"
#include "../include/Plasma.h"
#include "../include/PlasmaTypes.h"
#include "../include/ConfigParser/Properties.h"

//TODO: gpu cell in the global array at copy from there appears to be not initialized
using namespace std;

void printHelp();
PlasmaConfig initConfig(std::ifstream &is);
void deinitConfig(PlasmaConfig& config);

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
            conf = initConfig(myfile);
        } else {
            cout << "Unable to open file: " << config << endl;
            return -1;
        }

        if (conf.checkFile != NULL && !isFileExist(conf.checkFile)) {
            std::cerr << "Check file: " << conf.checkFile << " does not exist" << std::endl;
            return -1;
        }

        InitMPI(argc, argv);

        cout << "begin Particle size " <<  sizeof(Particle) << endl;

        try {
            plasma = new Plasma(&conf);

            if (conf.computeFromFile != NULL) {

                if (isFileExist(conf.computeFromFile)) {
                    plasma->Initialize(conf.computeFromFile);
                } else {
                    cout << "Unable to open file: " << conf.computeFromFile << endl;
                    return -1;
                }
            } else {
                plasma->Initialize();
            }


            plasma->Compute();

            delete plasma;
        } catch (std::bad_alloc &e) {
            std::cerr << "Unable to allocate memory" << std::endl;
        } catch (std::exception &e) {
            std::cout << "Unexpected termination of program. Explanatory string: " << e.what() << std::endl;
        }

        deinitConfig(conf);

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

PlasmaConfig initConfig(std::ifstream &is) {
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
        conf.lx = properties.stringToDouble(properties.getProperty("lx"));
        conf.ly = properties.stringToDouble(properties.getProperty("ly"));
        conf.lz = properties.stringToDouble(properties.getProperty("lz"));
        conf.lp = properties.stringToInt(properties.getProperty("lp"));
        conf.nx = properties.stringToInt(properties.getProperty("nx"));
        conf.ny = properties.stringToInt(properties.getProperty("ny"));
        conf.nz = properties.stringToInt(properties.getProperty("nz"));
        conf.tau = properties.stringToDouble(properties.getProperty("tau"));
        conf.beamPlasma = properties.stringToInt(properties.getProperty("beamPlasma"));
        conf.meh = properties.stringToInt(properties.getProperty("meh"));

/* unused now
        conf.externalMagnFieldX = properties.stringToDouble(properties.getProperty("externalMagnFieldX"));
        conf.px = properties.stringToDouble(properties.getProperty("px"));
        conf.py = properties.stringToDouble(properties.getProperty("py"));
        conf.bx = properties.stringToDouble(properties.getProperty("bx"));
        conf.by = properties.stringToDouble(properties.getProperty("by"));
        conf.bz = properties.stringToDouble(properties.getProperty("bz"));
*/
        conf.st = properties.stringToInt(properties.getProperty("st"));
        conf.lst = properties.stringToInt(properties.getProperty("lst"));
        conf.saveStep = properties.stringToInt(properties.getProperty("saveStep"));
        conf.startSave = properties.stringToInt(properties.getProperty("startSave"));
        conf.checkStep = properties.stringToInt(properties.getProperty("checkStep"));
        std::string checkFile = properties.getProperty("checkFile");
        if (checkFile.empty()) {
            conf.checkFile = NULL;
        }
        else {
            int l = (int)checkFile.length() + 1;
            conf.checkFile = new char[l + 1];
            memcpy((void*)conf.checkFile, (void*)checkFile.c_str(), (size_t)l + 1);
        }
        std::string computeFile = properties.getProperty("computeFromFile");
        if (computeFile.empty()) {
            conf.computeFromFile = NULL;
        }
        else {
            int l = (int)computeFile.length() + 1;
            conf.computeFromFile = new char[l + 1];
            memcpy((void*)conf.computeFromFile, (void*)computeFile.c_str(), (size_t)l + 1);
        }
    }
    catch (std::exception &e) {
        std::cout << "Config parsing error: " << e.what() << std::endl;
        exit(-1);
    }

    return conf;
}

void deinitConfig(PlasmaConfig& config) {
    delete[] config.checkFile;
}

bool isFileExist(const char * name) {
    ifstream f(name);
    return f.good();
}