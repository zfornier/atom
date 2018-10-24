LD=mpicxx
CUDACC=nvcc
CXX=mpicxx
CPP=g++
CUDA=/usr/local/cuda
CUDALIB=$(CUDA)/lib64
SRCDIR := src

LDFLAGS= -lm -L$(CUDALIB) -arch=sm_52
#CUDAFLAGS= --maxrregcount=128  -arch=sm_52 --ptxas-options=-v -I/usr/local/cuda-7.5/include/
CUDAFLAGS= -arch=sm_52 -lineinfo --maxrregcount=128 -g -I$(CUDA)/include/ #--relocatable-device-code=true

#LDFLAGS= -lm -L$(CUDALIB) -arch=sm_61
##CUDAFLAGS= --maxrregcount=128  -arch=sm_35 --ptxas-options=-v -I/usr/local/cuda-7.5/include/
#CUDAFLAGS= -arch=sm_61 -lineinfo --maxrregcount=128 -g -I$(CUDA)/include/ #--relocatable-device-code=true

CUDALIBS=  -g -L$(CUDALIB) -lcuda -lcudart #-lthrust
MPIFLAGS=
CFLAGS=

OBJ = main.o mpi_shortcut.o service_functions.o compare.o maxwell.o load_data.o archAPI.o
#plasma.o

all: $(OBJ)
	$(LD) -g -o $@ $^ $(CFLAGS) $(DBFLAGS) $(CUDALIBS)

main.o: $(SRCDIR)/main.cu
%.o: $(SRCDIR)/%.cu
	$(CUDACC) -g -c -o $@ $< $(CUDAFLAGS)

%.o: $(SRCDIR)/%.cxx
	$(CXX) -g -c -o $@ $< $(MPIFLAGS)

%.o: $(SRCDIR)/%.cpp
	$(CPP) -g -c -o $@ $< $(CBFLAGS)

clean:
	rm *.o all