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
    create_vector(curr_vector, 1000);
    return 0;
}