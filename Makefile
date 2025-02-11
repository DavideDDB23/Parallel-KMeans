#
# K-means 
#
# Parallel computing (Degree in Computer Engineering)
# 2022/2023
#
# (c) 2023 Diego Garcia-Alvarez and Arturo Gonzalez-Escribano
# Grupo Trasgo, Universidad de Valladolid (Spain)
#

# Compilers
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
    CC = gcc-14
    OMPFLAG = -fopenmp
    MPICC = mpicc
    CUDACC = nvcc
    PLATFORM_FLAGS = -isysroot $(shell xcrun --show-sdk-path)
else
    CC = gcc -mavx2
    OMPFLAG = -fopenmp
    MPICC = mpicc
    CUDACC = nvcc
	CUDAFLAGS=--generate-line-info -arch=sm_75
endif

# Flags for optimization and libs
FLAGS = -O3 -Wall -g
LIBS = -lm

# Targets to build
OBJS=KMEANS_seq.out KMEANS_omp.out KMEANS_mpi.out KMEANS_cuda.out KMEANS_omp_mpi.out

# Rules. By default show help
help:
	@echo
	@echo "K-means clustering method"
	@echo "Group Trasgo, Universidad de Valladolid (Spain)"
	@echo
	@echo "make KMEANS_seq         Build only the sequential version"
	@echo "make KMEANS_omp         Build only the OpenMP version"
	@echo "make KMEANS_mpi         Build only the MPI version"
	@echo "make KMEANS_cuda        Build only the CUDA version"
	@echo "make KMEANS_omp_mpi     Build the MPI+OMP version"
	@echo
	@echo "make all                Build all versions (Sequential, OpenMP, etc.)"
	@echo "make debug              Build all versions with demo output for small surfaces"
	@echo "make clean              Remove targets"
	@echo

# Build all versions
all: $(OBJS)

# Target rules
KMEANS_seq: KMEANS.c
	$(CC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@.out

KMEANS_omp: KMEANS_omp.c
	$(CC) $(FLAGS) $(DEBUG) $(OMPFLAG) $< $(LIBS) -o $@.out

KMEANS_mpi: KMEANS_mpi.c
	$(MPICC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@.out

KMEANS_cuda: KMEANS_cuda.cu
	$(CUDACC) $(DEBUG) $(CUDAFLAGS) $< $(LIBS) -o $@.out

KMEANS_omp_mpi: KMEANS_omp_mpi.c
	$(MPICC) $(FLAGS) $(DEBUG) $(OMPFLAG) $< $(LIBS) -o $@.out

# Clean command
clean:
	rm -rf $(OBJS)

# Compile in debug mode
debug:
	make DEBUG="-DDEBUG -g" FLAGS= all