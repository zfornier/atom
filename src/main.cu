#include "../include/gpu_plasma.h"
#include <stdlib.h>
#include "../include/mpi_shortcut.h"
#include "ConfigParser/Properties.h"

//TODO: gpu cell in the global array at copy from there appears to be not initialized
using namespace std;

typedef struct {
    double tempX;                // plasma electron temperature along X
    double tempY;                // plasma electron temperature along Y
    double tempZ;                // plasma electron temperature along Z
    double beamImp;              // beam impulse
    double beamVelDisp;          // beam velocity dispersion
    double beamPlasmaDensityRat; // beam and plasma density ratio
    double plsmDensity;          // plasma density
    double externalMagnFieldX;   // external magnetic field (along X)
    double lx;                   // domain size X
    double ly;                   // domain size Y
    double lz;                   // domain size Z
    double px;                   // plasma size X
    double py;                   // plasma size Y
    double bx;                   // beam size X
    double by;                   // beam size Y
    double bz;                   // beam size Z
    int lp;                      // average number of particles in cell
    int nx;                      // number of mesh nodes along X
    int ny;                      // number of mesh nodes along Y
    int nz;                      // number of mesh nodes along Z
    double tau;                  // timestep
    int beamPlasma;              // 1 if beam-plasma interaction, 0 if beam-beam
    int startFromFile;           // moment to start from saved
    int phase;                   // phase to start from save
    int ts;                      // total steps
    int ms;                      // number of steps between diagnostic files
    int nsteps;                  //
    int saveStep;                // save every saveStep step
    int startSave;               // start save from startSave step
    int checkOnStep;             // check on checkOnStep step
    std::string checkFile;             // file to check with

} Config;

Config readConfig(std::ifstream &is) {
   Properties properties;
   properties.load(is);

   Config conf = Config();
   try {
      conf.tempX = properties.stringToDouble(properties.getProperty("tempX"));
      conf.tempY = properties.stringToDouble(properties.getProperty("tempY"));
      conf.tempZ = properties.stringToDouble(properties.getProperty("tempZ"));
      conf.beamImp = properties.stringToDouble( properties.getProperty("beamImp"));
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
      conf.checkOnStep = properties.stringToInt(properties.getProperty("checkOnStep"));
      conf.checkFile = properties.getProperty("checkFile");
   }
   catch (const std::invalid_argument& e) {
      throw std::exception("Error while getting properties: " + e.what());
   }
   catch (const std::out_of_range& e) {
      throw std::exception("Error while getting properties: " + e.what());
   }

   return conf;
}

int main(int argc,char*argv[]) {
   Plasma *plasma;

   char* config = NULL;
   int c;

   while ((c = getopt (argc, argv, "i:")) != -1){
      switch(c) {
         case 'i':
            config = optarg;
            break;
         default:
            break;
      }
   }

   if(config != NULL) {
      string line;
      ifstream myfile(config);
      Config conf;

      if (myfile.is_open()) {
         conf = readConfig(myfile);
         std::cout << conf.tempX << std::endl;
         std::cout << conf.tempY << std::endl;
         std::cout << conf.tempZ << std::endl;
         std::cout << conf.beamImp << std::endl;
         std::cout << conf.beamVelDisp << std::endl;
         std::cout << conf.beamPlasmaDensityRat << std::endl;
         std::cout << conf.plsmDensity << std::endl;
         std::cout << conf.externalMagnFieldX << std::endl;
         std::cout << conf.lx << std::endl;
         std::cout << conf.ly << std::endl;
         std::cout << conf.lz << std::endl;
         std::cout << conf.px << std::endl;
         std::cout << conf.py << std::endl;
         std::cout << conf.bx << std::endl;
         std::cout << conf.by << std::endl;
         std::cout << conf.bz << std::endl;
         std::cout << conf.lp << std::endl;
         std::cout << conf.nx << std::endl;
         std::cout << conf.ny << std::endl;
         std::cout << conf.nz << std::endl;
         std::cout << conf.tau << std::endl;
         std::cout << conf.beamPlasma << std::endl;
         std::cout << conf.startFromFile << std::endl;
         std::cout << conf.phase << std::endl;
         std::cout << conf.ts << std::endl;
         std::cout << conf.ms << std::endl;
         std::cout << conf.nsteps << std::endl;
         std::cout << conf.saveStep << std::endl;
         std::cout << conf.startSave << std::endl;
         std::cout << conf.checkOnStep << std::endl;
         std::cout << conf.checkFile << std::endl;
      }
      else {
         cout << "Unable to open file: " <<  config << endl;
         return 0;
      }

      InitMPI(argc,argv);

      printf("begin Particle size %zu\n", sizeof(Particle));

      plasma = new Plasma(100,4,4,1.1424,0.05,0.05,1.0,2000,1.0,0.001);

      plasma->Initialize();

      plasma->Compute();

      CloseMPI();

      delete plasma;
   }
   else {
      printf("Config file is expected.\n");
   }

   return 0;
}
