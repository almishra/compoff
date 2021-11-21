#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <ctime>

#ifndef LA
#define LA 10
#endif
#ifndef LB
#define LB 1
#endif
#ifndef LC
#define LC 1
#endif
#ifndef LD
#define LD 1
#endif
#ifndef LE
#define LE 4
#endif
#ifndef M
#define M 3
#endif
#ifndef N
#define N 3
#endif

void matrix_vector( 
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LB][LC][LD][N])
{
  for(int a=0; a<LA; a++) {
    for(int b=0; b<LB; b++) {
      for(int c=0; c<LC; c++) {
        for(int d=0; d<LD; d++) {

          for(int n=0; n<N; n++) {
            vectorC[a][b][c][d][n] = 0;
          }
          for(int e=0; e<LE; e++) {
            float temp[N];
            for(int m=0; m<M; m++) {
              for(int n=0; n<N; n++) {
                temp[n] += matrixA[a][b][c][d][e][m][n] * vectorB[a][b][c][d][e][n];
              }
            }
            for(int n=0; n<N; n++) {
              int m = (e%2==0) ? 1 : -1;
              vectorC[a][b][c][d][n] += m * temp[n];
            }
          }
        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
}

void matrix_vector_off( 
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LA][LB][LC][LD][N],
    int dev)
{
  printf("DtoH: %lu\n", 
      sizeof(float)*LA*LB*LC*LD*LE*M*N + 
      sizeof(float)*LA*LB*LC*LD*LE*M);
  printf("HtoD: %lu\n", 
      sizeof(float)*LA*LB*LC*LD*M);
#pragma omp target teams distribute parallel for collapse(4) \
  map(from:vectorC[dev][0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to:matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to:vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N]) device(dev)
  for(int a=0; a<LA; a++) {
    for(int b=0; b<LB; b++) {
      for(int c=0; c<LC; c++) {
        for(int d=0; d<LD; d++) {

          for(int n=0; n<N; n++) {
            vectorC[dev][a][b][c][d][n] = 0;
          }
          for(int e=0; e<LE; e++) {
            float temp[N];
            for(int m=0; m<M; m++) {
              for(int n=0; n<N; n++) {
                temp[n] += matrixA[a][b][c][d][e][m][n] * vectorB[a][b][c][d][e][n];
              }
            }
            for(int n=0; n<N; n++) {
              int m = (e%2==0) ? 1 : -1;
              vectorC[dev][a][b][c][d][n] += m * temp[n];
            }
          }
        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
}

int main() {

  float (*matA)[LB][LC][LD][LE][M][N] =
    (float (*)[LB][LC][LD][LE][M][N]) malloc(sizeof(float)*LA*LB*LC*LD*LE*M*N);
  for(int a=0; a<LA; a++)
    for(int b=0; b<LB; b++)
      for(int c=0; c<LC; c++)
        for(int d=0; d<LD; d++)
          for(int e=0;e<LE;e++)
            for(int m=0;m<M;m++)
              for(int n=0;n<N;n++)
                  matA[a][b][c][d][e][m][n] = 0.22;

  float (*vecB)[LB][LC][LD][LE][N] =
    (float (*)[LB][LC][LD][LE][N]) malloc(sizeof(float)*LA*LB*LC*LD*LE*N);
  for(int a=0; a<LA; a++)
    for(int b=0; b<LB; b++)
      for(int c=0; c<LC; c++)
        for(int d=0; d<LD; d++)
          for(int e=0;e<4;e++)
            for(int n=0;n<3;n++)
                vecB[a][b][c][d][e][n] = 0.001;

  float (*vecCPU)[LB][LC][LD][N] =
    (float (*)[LB][LC][LD][N]) malloc(sizeof(float)*LA*LB*LC*LD*N);
  matrix_vector(matA, vecB, vecCPU);
  for(int a=0; a<LA; a++)
    for(int b=0; b<LB; b++)
      for(int c=0; c<LC; c++)
        for(int d=0; d<LD; d++) {
          for(int n=0;n<N; n++)
            printf("%lf ", vecCPU[a][b][c][d][n]);
          printf("\n");
        }

#if OFF
  int dev = omp_get_num_devices();
  if(dev == 0) {
    printf("No device available\n");
    exit(-1);
  }
  float (*vecC)[LA][LB][LC][LD][N] =
    (float (*)[LA][LB][LC][LD][N]) malloc(sizeof(float)*dev*LA*LB*LC*LD*N);

  printf("Here\n");
  fflush(stdout);
//#pragma omp parallel for
  for(int device = 0; device < dev; device++) {
    printf("Offloading on GPU %d\n", device);
    matrix_vector_off(matA, vecB, vecC, device);

    for(int a=0; a<LA; a++) {
      for(int b=0; b<LB; b++) {
        for(int c=0; c<LC; c++) {
          for(int d=0; d<LD; d++) {
            for(int n=0; n<N; n++) {
              if(vecCPU[a][b][c][d][n] != vecC[device][a][b][c][d][n]) {
                printf("FAIL\n");
                exit(-1);
              }
            }
          }
        }
      }
    }
  }
  printf("PASS\n");
#endif
  return 0;
}
