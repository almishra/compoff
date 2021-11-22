#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
#include <unistd.h>

#ifndef COLLAPSE
#define COLLAPSE 2
#endif
#ifndef LA
#define LA 1
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
#define LE 3
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
  omp_set_num_threads(28);
#pragma omp parallel for
  for(int a=0; a<LA; a++) {
    for(int b=0; b<LB; b++) {
      for(int c=0; c<LC; c++) {
        for(int d=0; d<LD; d++) {

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
          } // end e loop

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
#pragma omp target teams distribute parallel for collapse(COLLAPSE) \
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
            for(int n=0; n<N; n++) {
              temp[n] = 0;
              for(int m=0; m<M; m++) {
                temp[n] += matrixA[a][b][c][d][e][m][n] * vectorB[a][b][c][d][e][n];
              }
            }
            for(int n=0; n<N; n++) {
              int m = (e%2==0) ? 1 : -1;
              vectorC[dev][a][b][c][d][n] += m * temp[n];
            }
          } // end e loop

        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
}

int main() 
{
  printf("Device,LA,LB,LC,LD,LE,M,N,runtime(us),runtime(s)\n");
#ifdef DEBUG
  fprintf(stderr, "Total memory for matA = %lf\n", sizeof(float)*LA*LB*LC*LD*LE*M*N / 1024.0 / 1024.0 / 1024.0);
  fprintf(stderr, "Total memory for vecB = %lf\n", sizeof(float)*LA*LB*LC*LD*LE*N / 1024.0 / 1024.0 / 1024.0);
  fprintf(stderr, "Total memory for vecC = %lf\n", sizeof(float)*LA*LB*LC*LD*N / 1024.0 / 1024.0 / 1024.0);
#endif
  float (*matA)[LB][LC][LD][LE][M][N] =
    (float (*)[LB][LC][LD][LE][M][N]) malloc(sizeof(float)*LA*LB*LC*LD*LE*M*N);
  for(int a=0; a<LA; a++)
    for(int b=0; b<LB; b++)
      for(int c=0; c<LC; c++)
        for(int d=0; d<LD; d++)
          for(int e=0;e<LE;e++)
            for(int m=0;m<M;m++)
              for(int n=0;n<N;n++)
                  matA[a][b][c][d][e][m][n] = 0.1;

  float (*vecB)[LB][LC][LD][LE][N] =
    (float (*)[LB][LC][LD][LE][N]) malloc(sizeof(float)*LA*LB*LC*LD*LE*N);
  for(int a=0; a<LA; a++)
    for(int b=0; b<LB; b++)
      for(int c=0; c<LC; c++)
        for(int d=0; d<LD; d++)
          for(int e=0;e<LE;e++)
            for(int n=0;n<N;n++)
                vecB[a][b][c][d][e][n] = 2.5;

  float (*vecCPU)[LB][LC][LD][N] =
    (float (*)[LB][LC][LD][N]) malloc(sizeof(float)*LA*LB*LC*LD*N);

  struct timeval  tv1_cpu, tv2_cpu;
  gettimeofday(&tv1_cpu, NULL);
  matrix_vector(matA, vecB, vecCPU);
  gettimeofday(&tv2_cpu, NULL);
  long long runtime_cpu = (tv2_cpu.tv_sec - tv1_cpu.tv_sec) * 1000000;
  runtime_cpu += tv2_cpu.tv_usec - tv1_cpu.tv_usec;
  printf("CPU,%d,%d,%d,%d,%d,%d,%d,%lld,%0.4f\n", LA,LB,LC,LD,LE,M,N,runtime_cpu, (double)(runtime_cpu/1000000.0));
  fflush(stdout);

#ifdef DEBUG1
  for(int a=0; a<LA; a++)
    for(int b=0; b<LB; b++)
      for(int c=0; c<LC; c++)
        for(int d=0; d<LD; d++) {
          for(int n=0;n<N; n++)
            fprintf(stderr, "%lf ", vecCPU[a][b][c][d][n]);
          fprintf(stderr, "\n");
        }
  fflush(stderr);
#endif

#ifdef OFF
  int dev = omp_get_num_devices();
  if(dev == 0) {
    fprintf(stderr, "No device available\n");
    exit(-1);
  }
  float (*vecC)[LA][LB][LC][LD][N] =
    (float (*)[LA][LB][LC][LD][N]) malloc(sizeof(float)*dev*LA*LB*LC*LD*N);

#pragma omp parallel for
  for(int device = 0; device < dev; device++) {
#ifdef DEBUG
    fprintf(stderr, "Offloading on GPU %d\n", device);
#endif
    struct timeval  tv1, tv2;
    gettimeofday(&tv1, NULL);
    matrix_vector_off(matA, vecB, vecC, device);
    gettimeofday(&tv2, NULL);
    long long runtime = (tv2.tv_sec - tv1.tv_sec) * 1000000;
    runtime += tv2.tv_usec - tv1.tv_usec;
#pragma omp critical
    {
      printf("GPU%d,%d,%d,%d,%d,%d,%d,%d,%lld,%0.4f\n", device, LA,LB,LC,LD,LE,M,N,runtime, (double)(runtime/1000000.0));
      fflush(stdout);
    }

#ifdef DEBUG1
#pragma omp critical
    for(int a=0; a<LA; a++)
      for(int b=0; b<LB; b++)
        for(int c=0; c<LC; c++)
          for(int d=0; d<LD; d++) {
            for(int n=0;n<N; n++)
              fprintf(stderr, "%lf ", vecC[device][a][b][c][d][n]);
            printf("\n");
          }
    fflush(stderr);
#endif
    for(int a=0; a<LA; a++) {
      for(int b=0; b<LB; b++) {
        for(int c=0; c<LC; c++) {
          for(int d=0; d<LD; d++) {
            for(int n=0; n<N; n++) {
              if(vecCPU[a][b][c][d][n] != vecC[device][a][b][c][d][n]) {
                printf("FAILED for %d in device %d\n", n, device);
              }
            }
          }
        }
      }
    }
  }
  fprintf(stderr, "PASS\n");
#endif
  return 0;
}
