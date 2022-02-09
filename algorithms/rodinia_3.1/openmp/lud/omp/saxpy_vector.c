#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

#include "common.h"
#ifdef OMP_OFFLOAD
#pragma offload_attribute(push, target(mic))
#endif


#define a(i) vector1[i]
#define b(i) vector2[i]
#define c(i) res[i]

void create_vector(float** vector, int dim) {
    float* curr = (float*) malloc(sizeof(float)*dim);
    int i;
    #pragma omp parallel for shared(curr) private(i)
    for(i=0;i<dim;i++){
        curr[i] = ((float) rand()/(float)(RAND_MAX));
    }
    *vector = curr;
}

void saxpy(float* vector1, float* vector2, int size){
    stopwatch sw;
    int i;
    stopwatch_start(&sw);
    #ifdef OMP_OFFLOAD
    #pragma omp target teams distribute parallel for simd \
        num_teams(10) map(to:vector1[0:size]) map(tofrom:vector2[0:size])
    #endif
    for(i=0;i<size;i++){
        b(i) = a(i)*b(i);
    }
    stopwatch_stop(&sw);
    printf("Time consumed(ms): %lf\n", 1000*get_interval_by_sec(&sw));
}
int main(int argc, char* argv[]){
    float* vector1;
    float* vector2;
    int i, vector_dim = 1000;
    create_vector(&vector1, vector_dim);
    create_vector(&vector2, vector_dim);
    saxpy(vector1, vector2, vector_dim);
    return 0;
}