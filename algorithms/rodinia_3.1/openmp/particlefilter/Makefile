#makefile

gpu:
	gcc -fopenmp -foffload=nvptx-none ex_particle_OPENMP_seq.c -o particle_filter_gpu -lm 

openmp: ex_particle_OPENMP_seq.c
	gcc -O3 -ffast-math -fopenmp ex_particle_OPENMP_seq.c -o particle_filter -lm 


icc_openmp: ex_particle_OPENMP_seq.c
	icc -O3 -fast -openmp ex_particle_OPENMP_seq.c -o particle_filter


clean:
	rm particle_filter particle_filter_gpu
