CC = clang++
SRC = pathfinder.cpp
EXE = pathfinder
FLAGS = -fopenmp -O3 -L/lustre/projects/compoff-group/opt/llvm/lib -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70 

release:
	$(CC) $(SRC) $(FLAGS) -o $(EXE)

debug:
	$(CC) $(SRC) -g -Wall -o $(EXE)

clean:
	rm -f pathfinder


