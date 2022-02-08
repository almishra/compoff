#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"

#define A(_i,_j) a[_i*matrix_dim+_j]
#define B(_i,_j) b[_i*matrix_dim+_j]
#define C(_i,_j) c[_i*matrix_dim+_j]

void saxpy(float* a, float* b, float* c, int matrix_dim){
    int i;
    for(i=0;i<matrix_dim;i++){
        int j;
        for(j=0;j<matrix_dim;j++){
            float totalSum = 0.0f;
            int k;
            for(k=0;k<matrix_dim;k++)
                totalSum+=A(i,k)*B(k,j);
            C(i,j) = totalSum;
        }
    }
}

int main(int argc, char* argv[]){
    int matrix_dim = 32;
    float *mat1, *mat2, *res;
    int ret1 = create_matrix(&mat1, matrix_dim);
    int ret2 = create_matrix(&mat2, matrix_dim);
    int ret3 = create_matrix(&res, matrix_dim);
    if(ret1!=RET_SUCCESS || ret2!=RET_SUCCESS || ret3!=RET_SUCCESS){
        exit(0);
    }
    saxpy(mat1, mat2, mat3, matrix_dim);
    return 0;
}