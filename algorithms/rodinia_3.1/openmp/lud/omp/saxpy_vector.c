#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

void create_vector(float* vector, int dim) {
    vector = (float*) malloc(sizeof(float)*dim);
    int i;
    #pragma omp parallel for shared(vector) private(i)
    for(i=0;i<dim;i++){
        vector[i] = ((float) rand()/(float)(RAND_MAX));
    }
}

int main(int argc, char* argv[]){
    float* curr_vector;
    int i, vector_dim = 10;
    create_vector(curr_vector, vector_dim);
    #pragma omp parallel for
    for(i=0;i<10;i++)
        printf("%.6f ", curr_vector[i]);
    printf("\n");
    return 0;
}