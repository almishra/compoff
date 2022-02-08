#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"

#define A(_i,_j) a[_i*matrix_dim+_j]
#define B(_i,_j) b[_i*matrix_dim+_j]
#define C(_i,_j) c[_i*matrix_dim+_j]

void saxpy_normal(float* a, float* b, float* c, int matrix_dim){
    int i,j,k;
    stopwatch sw;
    stopwatch_start(&sw);
    for(i=0;i<matrix_dim;i++){
        for(j=0;j<matrix_dim;j++){
            float totalSum = 0.0f;
            for(k=0;k<matrix_dim;k++)
                totalSum+=A(i,k)*B(k,j);
            C(i,j) = totalSum;
        }
    }
    stopwatch_stop(&sw);
    printf("Time consumed(ms): %lf\n", 1000*get_interval_by_sec(&sw));
}

void saxpy_omp(float* a, float* b, float* c, int matrix_dim){
    int i,j,k;
    stopwatch sw;
    stopwatch_start(&sw);
    int num_threads;
    #pragma omp parallel for
    for(i=0;i<matrix_dim;i++){
        num_threads = omp_get_num_threads();
        for(j=0;j<matrix_dim;j++){
            float totalSum = 0.0f;
            for(k=0;k<matrix_dim;k++){
                totalSum+=A(i,k)*B(k,j);
            }
            C(i,j) = totalSum;
        }
    }
    stopwatch_stop(&sw);
    printf("Time consumed(ms): %lf\n", 1000*get_interval_by_sec(&sw));
    printf("Num Threads: %d\n", num_threads);
}
int main(int argc, char* argv[]){
    int matrix_dim = 1000;
    float *mat1, *mat2, *res;
    int ret1 = create_matrix(&mat1, matrix_dim);
    int ret2 = create_matrix(&mat2, matrix_dim);
    int ret3 = create_matrix(&res, matrix_dim);
    if(ret1!=RET_SUCCESS || ret2!=RET_SUCCESS || ret3!=RET_SUCCESS){
        exit(0);
    }
    saxpy_normal(mat1, mat2, res, matrix_dim);
    saxpy_omp(mat1, mat2, res, matrix_dim);
    return 0;
}