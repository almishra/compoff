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
    printf("matrix_dim,mem_to,mem_from,runtime(us), runtime(s)");
    for(;matrix_dim<=1000;){
        float *m;
        //stopwatch sw;
        int ret = create_matrix(&m, matrix_dim);
        if (ret != RET_SUCCESS) {
            m = NULL;
            fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
        }
        // stopwatch_start(&sw);
        // lud_omp(m, matrix_dim);
        // stopwatch_stop(&sw);
        // int usecs = get_interval_by_usec(&sw);
        // double secnds = get_interval_by_sec(&sw);
        // int size = matrix_dim;
        // printf("%d,%lu,%lu,%d,%f\n",size, (size*(size+1))*sizeof(int), size*size*sizeof(int), \
        //         usecs, secnds);
        matrix_dim*=10;
    }
}