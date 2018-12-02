#include "../include/gpu_plasma.h"
#include <stdlib.h>
#include "../include/mpi_shortcut.h"
#include "ConfigParser/Properties.h"

//TODO: gpu cell in the global array at copy from there appears to be not initialized

typedef struct {
    double tempX;                // plasma electron temperature along X
    double tempY;                // plasma electron temperature along Y
    double tempX;                // plasma electron temperature along Z
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
    int save_step;               // save every saveStep step
    int startSave;               // start save from startSave step
    int checkOnStep;             // check on checkOnStep step
    char* checkFile;             // file to check with

} Config;

Config readConfig(std::ifstream &is) {
   Config conf;

   // read props

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
