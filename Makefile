SEQSOURCE = ballAlg.c	
SEQTARGET = ballAlg
PARSOURCE = ballAlg-omp.c	
PARTARGET = ballAlg-omp
MPISOURCE = ballAlg-mpi.c
MPITARGET = ballAlg-mpi
CC = gcc
MPICC = mpicc
MPIGLAGS = -fopenmp -O3 -lm
FLAGS = -fopenmp -O3 -lm

all: ballAlg

ballAlg: ballAlg.c ballAlg-omp.c ballAlg-mpi.c
	$(CC) $(SEQSOURCE) -o $(SEQTARGET) $(FLAGS)
	$(CC) $(PARSOURCE) -o $(PARTARGET) $(FLAGS)
	$(MPICC) $(MPISOURCE)  -o $(MPITARGET) $(MPIGLAGS)
