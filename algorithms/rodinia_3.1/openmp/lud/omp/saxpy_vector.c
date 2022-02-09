#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

void create_vector(float** vector, int dim) {
    float* curr = (float*) malloc(sizeof(float)*dim);
    int i;
    #pragma omp parallel for shared(curr) private(i)
    for(i=0;i<dim;i++){
        curr[i] = ((float) rand()/(float)(RAND_MAX));
    }
    *vector = curr;
}

int main(int argc, char* argv[]){
    float* curr_vector;
    int i, vector_dim = 10;
    create_vector(curr_vector, vector_dim);
    for(i=0;i<vector_dim;i++)
        printf("%.6f ", curr_vector[i]);
    printf("\n");
    return 0;
}