#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"

void saxpy(float* a, float* b){
}
int main(int argc, char* argv[]){
    int matrix_dim = 32;
    float *mat1, *mat2;
    int ret1 = create_matrix(&mat1, matrix_dim);
    int ret2 = create_matrix(&mat2, matrix_dim);
    if(ret1!=RET_SUCCESS || ret2!=RET_SUCCESS){
        exit(0);
    }
    // for(i=0;i<matrix_dim;i++){
    //     int ii;
    //     for(ii=0;ii<matrix_dim;ii++){
    //         printf("%.6f ", mat[i*matrix_dim+ii]);
    //     }
    //     printf("\n");
    // }
    return 0;
}