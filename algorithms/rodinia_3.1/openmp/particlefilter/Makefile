#makefile

gpu:
	clang -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_TOOLKIT_ROOT_DIR} -Xopenmp-target -march=sm_60 ex_particle_OPENMP_seq.c -o particle_filter_gpu -lm 

openmp: ex_particle_OPENMP_seq.c
	gcc -O3 -ffast-math -fopenmp ex_particle_OPENMP_seq.c -o particle_filter -lm 


icc_openmp: ex_particle_OPENMP_seq.c
	icc -O3 -fast -openmp ex_particle_OPENMP_seq.c -o particle_filter


clean:
	rm particle_filter particle_filter_gpu
