#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"

extern void
lud_omp(float *m, int matrix_dim);

int main ( int argc, char *argv[]){
    int matrix_dim = 100;
    float *m;
    for(;matrix_dim<=1000;){
        int ret = create_matrix(&m, matrix_dim);
        if (ret != RET_SUCCESS) {
            m = NULL;
            fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
        }
        lud_omp(m, matrix_dim);
        matrix_dim*=10;
    }
}