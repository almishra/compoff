#include "matrix_vector.h"

void matrix_vector_variant6( 
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LB][LC][LD][N])
{
#ifdef MEMCPY
#pragma omp target enter data \
  map(alloc: vectorC[0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to: matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to: vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N])
#endif
#pragma omp target teams distribute parallel for collapse(3) schedule(guided)
  for(int a=0; a<LA; a++) {
    for(int b=0; b<LB; b++) {
      for(int c=0; c<LC; c++) {
        for(int d=0; d<LD; d++) {

          for(int count = 0; count <COUNT; count++) {
          for(int n=0; n<N; n++) {
            vectorC[a][b][c][d][n] = 0;
          }
          for(int e=0; e<LE; e++) {
            float temp[N];
            for(int n=0; n<N; n++) {
              temp[n] = 0;
              for(int m=0; m<M; m++) {
                temp[n] += matrixA[a][b][c][d][e][m][n] * vectorB[a][b][c][d][e][n];
              }
            }
            for(int n=0; n<N; n++) {
              int m = (e%2==0) ? 1 : -1;
              vectorC[a][b][c][d][n] += m * temp[n];
            }
          }
          }

        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
#ifdef MEMCPY
#pragma omp target exit data \
  map(from: vectorC[0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(delete: matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(delete: vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N])
#endif
}
