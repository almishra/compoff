#!/bin/bash
#SBATCH -p milan-64core
#SBATCH --time=01-00:00:00
#SBATCH --job-name=first-test
#SBATCH --output=/lustre/home/shmmohammed/bfs.log
#SBATCH --mail-user=sharfuddin.mohammed@stonybrook.edu

module load /lustre/projects/compoff-group/modulefiles/clang/14.0.0
cd projects/compoff/algorithms/rodinia_3.1/openmp/bfs
make gpu
for i in {1000..100000..1000}
  do
    ./bfs_gpu $i >> /lustre/home/shmmohammed/projects/compoff/benchmarkdata/bfs.csv
 done