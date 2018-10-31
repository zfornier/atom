#include "../include/gpu_plasma.h"
#include <stdlib.h>
#include "../include/mpi_shortcut.h"
#include "NetCdf/read_file.cpp"
//TODO: gpu cell in the global array at copy from there appears to be not initialized

int main(int argc,char*argv[])
{
//   Plasma *plasma;
//
//   InitMPI(argc,argv);
//
//   printf("begin Particle size %d \n", sizeof(Particle));
//
//   plasma = new Plasma(100,4,4,1.1424,0.05,0.05,1.0,2000,1.0,0.001);
//
//   plasma->Initialize();
//
//   plasma->Compute();
//
//   CloseMPI();
//
//   delete plasma;
//
   int result;

   char* name_file_in = NULL;
   char* name_file_out = NULL;
   int c;

   while ((c = getopt (argc, argv, "i:o:")) != -1){
      switch(c)
      {
         case 'i':
            name_file_in = optarg;
            break;
         case 'o':
            name_file_out = optarg;
            break;
      }
   }

   result = copyFile(name_file_in, name_file_out);

   cout << "File exported in NetCDF format with success" << endl;

   return 0;
}
