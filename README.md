# atom
Optimization pf Gurthang code to get rid of atomic operations. Next step: get current sum out of AccumulateCurrentWithParticlesInCell into separate kernel and parallelize it.

CMake:
* Добавлена первая версия автоматизированной сборки (довольно топорная).
  TODO: 
    - добавить флаги
    - настройки сборки
    
NetCFD:
* Добавлены файлы в проект

CMake:
mkdir build;cd build
NETCDF_CXX_DIR=../netcdf-cxx4/cxx4/ NETCDF_CXX_LIB_DIR=../netcdf-cxx4/build/cxx4/ cmake ../