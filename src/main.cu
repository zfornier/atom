#include "../include/gpu_plasma.h"
#include <stdlib.h>
#include "../include/mpi_shortcut.h"
#include "ConfigParser/Properties.h"

//TODO: gpu cell in the global array at copy from there appears to be not initialized

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
    char* checkFile;             // file to check with

} Config;

Config readConfig(std::ifstream &is) {
   Properties properties;
   properties.load(myfile);

   Config conf;
   conf.tempX = properties.getProperty("tempX");
   conf.tempY = properties.getProperty("tempY");
   conf.tempZ = properties.getProperty("tempZ");
   conf.beamImp = properties.getProperty("tempZ");
   conf.beamVelDisp = properties.getProperty("tempZ");
   conf.beamPlasmaDensityRat = properties.getProperty("");
   conf.plsmDensity = properties.getProperty("");
   conf.externalMagnFieldX = properties.getProperty("");
   conf.lx = properties.getProperty("lx");
   conf.ly = properties.getProperty("ly");
   conf.lz = properties.getProperty("lz");
   conf.px = properties.getProperty("px");
   conf.py = properties.getProperty("py");
   conf.bx = properties.getProperty("bx");
   conf.by = properties.getProperty("by");
   conf.bz = properties.getProperty("bz");
   conf.lp = properties.getProperty("lp");
   conf.nx = properties.getProperty("nx");
   conf.ny = properties.getProperty("ny");
   conf.nz = properties.getProperty("nz");
   conf.tau = properties.getProperty("tau");
   conf.beamPlasma = properties.getProperty("beamPlasma");
   conf.startFromFile = properties.getProperty("startFromFile");
   conf.phase = properties.getProperty("phase");
   conf.ts = properties.getProperty("ts");
   conf.ms = properties.getProperty("ms");
   conf.nsteps = properties.getProperty("nsteps");
   conf.saveStep = properties.getProperty("saveStep");
   conf.startSave = properties.getProperty("startSave");
   conf.checkOnStep = properties.getProperty("checkOnStep");
   conf.checkFile = properties.getProperty("checkFile");

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
//         std::cout << conf.tempX << std::endl;
//         std::cout << conf.tempY << std::endl;
//         std::cout << conf.tempZ << std::endl;
//         std::cout << conf.beamImp << std::endl;
//         std::cout << conf.beamVelDisp << std::endl;
//         std::cout << conf.beamPlasmaDensityRat << std::endl;
//         std::cout << conf.plsmDensity << std::endl;
//         std::cout << conf.externalMagnFieldX << std::endl;
//         std::cout << conf.lx << std::endl;
//         std::cout << conf.ly << std::endl;
//         std::cout << conf.lz << std::endl;
//         std::cout << conf.px << std::endl;
//         std::cout << conf.py << std::endl;
//         std::cout << conf.bx << std::endl;
//         std::cout << conf.by << std::endl;
//         std::cout << conf.bz << std::endl;
//         std::cout << conf.lp << std::endl;
//         std::cout << conf.nx << std::endl;
//         std::cout << conf.ny << std::endl;
//         std::cout << conf.nz << std::endl;
//         std::cout << conf.tau << std::endl;
//         std::cout << conf.beamPlasma << std::endl;
//         std::cout << conf.startFromFile << std::endl;
//         std::cout << conf.phase << std::endl;
//         std::cout << conf.ts << std::endl;
//         std::cout << conf.ms << std::endl;
//         std::cout << conf.nsteps << std::endl;
//         std::cout << conf.saveStep << std::endl;
//         std::cout << conf.startSave << std::endl;
//         std::cout << conf.checkOnStep << std::endl;
//         std::cout << conf.checkFile << std::endl;
      }
      else {
         cout << "Unable to open file: " <<  << endl;
         return 0;
      }

      InitMPI(argc,argv);

      printf("begin Particle size %d\n", sizeof(Particle));

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
