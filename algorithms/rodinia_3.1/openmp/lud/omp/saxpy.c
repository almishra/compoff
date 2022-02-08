#include <omp.h>
#include <stdio.h>

#include "common.h"

// void saxpy(float* a, float* b){
// }
int main(int argc, char* argv[]){
    int matrix_dim = 32;
    float *mat;
    int ret = create_matrix(&mat, matrix_dim);
    int i;
    for(i=0;i<matrix_dim;i++){
        int ii;
        for(ii=0;ii<matrix_dim;ii++){
            printf("%.6f ", mat[i*matrix_dim+ii]);
        }
        printf("\n");
    }
    return 0;
}