# atom
Optimization of Gurthang code to get rid of atomic operations. Next step: get current sum out of AccumulateCurrentWithParticlesInCell into separate kernel and parallelize it.

CMake:
* Добавлена возможность сборки через cmake.
  Опции:
  	* CMAKE\_BUILD\_TYPE: Release,Debug,Profiling,RelWithDebInfo,None
	* CUDA\_ARCH - one can specify compute cappability (20, 30, 35, etc.)
  Переменные окружения:
  	* NETCDF\_CXX\_DIR: Путь до заголовочных файлов NetCdf-cxx4
	* NETCDF\_CXX\_LIB\_DIR: Путь до библиотек NetCdf-cxx4

  пример использования:
  	`NETCDF_CXX_DIR=~/tmp/netcdf-cxx4/cxx4/ NETCDF_CXX_LIB_DIR=~/tmp/netcdf-cxx4/build/cxx4/ cmake -DCUDA_ARCH=52 -DCMAKE_BUILD_TYPE=Profiling ../`

NetCFD:
* Добавлены файлы в проект
