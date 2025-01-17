# GPU Compilation
CUFLAGS = modules/gpufuncs.o -L/usr/local/cuda/lib64 -lcudart
NVCCFLAGS = -ccbin g++   -m64  -gencode arch=compute_60,code=sm_60  -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86

MODES = -DFOURPCF -DPERIODIC -DDISCONNECTED
# Add the -DPERIODIC flag to run with periodic boundary conditions
# Add the -DDISCONNECTED flag to include the disconnected 4PCF contributions
# Add the -DFOURPCF flag to include the four-point correlator
# Add the -DFIVEPCF flag to include the five-point correlator
# Add the -DSIXPCF flag to include the six-point correlator

# FOR LINUX MACHINES WITH g++
CXX = g++ -std=c++0x -ffast-math -fopenmp -lgomp -Wall -pg -g
#Note - will have to figure out OPENMP vs CUDA - e.g. don't want 24 threads trying to each start a CUDA kernel
#CXXFLAGS = -O3 ${MODES} -DOPENMP # OPENMP multi-threading
CXXFLAGS = -O3 ${MODES} -DGPU ${CUFLAGS} # CUDA

AVX = -DAVX
# Remove this if you don't want AVX support

###############

default: encore encoreAVX

cpu: encore encoreAVX

gpu: gpufuncs encore encoreAVX

CMASM.o:
	$(CC) -DAVXMULTIPOLES generateCartesianMultipolesASM.c
	./a.out
	rm a.out
	$(CC) -Wall -c CMASM.c

encoreAVX: encore.cpp CMASM.o
	$(CXX) $(CXXFLAGS) $(AVX) encore.cpp CMASM.o -o encoreAVX

gpufuncs: 
	nvcc $(NVCCFLAGS) -c modules/gpufuncs.cu -o modules/gpufuncs.o

clean:
	$(RM) encore encoreAVX CMASM.o modules/gpufuncs.o
