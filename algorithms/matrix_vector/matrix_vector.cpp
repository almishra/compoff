#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <omp.h>
#include <sys/time.h>
#include <unistd.h>


enum VARIANT {
  STATIC1, STATIC2, STATIC3, STATIC4,
  DYNAMIC1, DYNAMIC2, DYNAMIC3, DYNAMIC4,
  GUIDED1, GUIDED2, GUIDED3, GUIDED4, 
  NUM_VAR
};

std::string VAR_STR[NUM_VAR] = {
  "static,1", "static,2", "static,3", "static,4",
  "dynamic,1", "dynamic,2", "dynamic,3", "dynamic,4",
  "guided,1", "guided,2", "guided,3", "guided,4"
};

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
#ifndef COUNT
#define COUNT 1000
#endif

static float (*vecCPU)[LB][LC][LD][N];
static long mem_to;
static long mem_from;
static int num_dev;

void print(float (*vecC)[LB][LC][LD][N])
{
#ifdef PRINT
#pragma omp critical
  for(int a=0; a<LA; a++)
    for(int b=0; b<LB; b++)
      for(int c=0; c<LC; c++)
        for(int d=0; d<LD; d++) {
          for(int n=0;n<N; n++)
            fprintf(stderr, "%lf ", vecC[a][b][c][d][n]);
          printf("\n");
        }
  fflush(stderr);
#endif
}

int validate(VARIANT var, float (*vecC)[LB][LC][LD][N], int device)
{
  for(int a=0; a<LA; a++)
    for(int b=0; b<LB; b++)
      for(int c=0; c<LC; c++)
        for(int d=0; d<LD; d++)
          for(int n=0; n<N; n++)
            if(vecCPU[a][b][c][d][n] != vecC[a][b][c][d][n]) {
              printf("FAILED on %d for %s in device %d\n", n, VAR_STR[var].c_str(), device);
              return -1;
            }
  return 0;
}

void output(VARIANT var, int device, long long runtime)
{
#pragma omp critical
  {
    printf("GPU%d,%s,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%0.4f\n", device, VAR_STR[var].c_str(), mem_to, mem_from, COUNT, LA,LB,LC,LD,LE,M,N,runtime, (double)(runtime/1000000.0));
    fflush(stdout);
  }
}

void matrix_vector_cpu( 
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
          } // end e loop
          }

        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
}

void matrix_vector_collapse1_static( 
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LA][LB][LC][LD][N],
    int dev)
{
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);
#pragma omp target teams distribute parallel for collapse(1) schedule(static) \
  map(from:vectorC[dev][0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to:matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to:vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N]) device(dev)
  for(int a=0; a<LA; a++) {
    for(int b=0; b<LB; b++) {
      for(int c=0; c<LC; c++) {
        for(int d=0; d<LD; d++) {

          for(int count = 0; count <COUNT; count++) {
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
          }

        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
  gettimeofday(&tv2, NULL);
  long long runtime = (tv2.tv_sec - tv1.tv_sec) * 1000000;
  runtime += tv2.tv_usec - tv1.tv_usec;
  output(STATIC1, dev, runtime);
}

void matrix_vector_collapse1_dynamic( 
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LA][LB][LC][LD][N],
    int dev)
{
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);
#pragma omp target teams distribute parallel for collapse(1) schedule(dynamic) \
  map(from:vectorC[dev][0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to:matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to:vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N]) device(dev)
  for(int a=0; a<LA; a++) {
    for(int b=0; b<LB; b++) {
      for(int c=0; c<LC; c++) {
        for(int d=0; d<LD; d++) {

          for(int count = 0; count <COUNT; count++) {
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
          }

        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
  gettimeofday(&tv2, NULL);
  long long runtime = (tv2.tv_sec - tv1.tv_sec) * 1000000;
  runtime += tv2.tv_usec - tv1.tv_usec;
  output(DYNAMIC1, dev, runtime);
}

void matrix_vector_collapse1_guided( 
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LA][LB][LC][LD][N],
    int dev)
{
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);
#pragma omp target teams distribute parallel for collapse(1) schedule(guided) \
  map(from:vectorC[dev][0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to:matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to:vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N]) device(dev)
  for(int a=0; a<LA; a++) {
    for(int b=0; b<LB; b++) {
      for(int c=0; c<LC; c++) {
        for(int d=0; d<LD; d++) {

          for(int count = 0; count <COUNT; count++) {
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
          }

        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
  gettimeofday(&tv2, NULL);
  long long runtime = (tv2.tv_sec - tv1.tv_sec) * 1000000;
  runtime += tv2.tv_usec - tv1.tv_usec;
  output(GUIDED1, dev, runtime);
}

void matrix_vector_collapse2_static( 
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LA][LB][LC][LD][N],
    int dev)
{
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);
#pragma omp target teams distribute parallel for collapse(2) schedule(static) \
  map(from:vectorC[dev][0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to:matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to:vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N]) device(dev)
  for(int a=0; a<LA; a++) {
    for(int b=0; b<LB; b++) {
      for(int c=0; c<LC; c++) {
        for(int d=0; d<LD; d++) {

          for(int count = 0; count <COUNT; count++) {
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
          }

        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
  gettimeofday(&tv2, NULL);
  long long runtime = (tv2.tv_sec - tv1.tv_sec) * 1000000;
  runtime += tv2.tv_usec - tv1.tv_usec;
  output(STATIC2, dev, runtime);
}

void matrix_vector_collapse2_dynamic( 
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LA][LB][LC][LD][N],
    int dev)
{
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);
#pragma omp target teams distribute parallel for collapse(2) schedule(dynamic) \
  map(from:vectorC[dev][0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to:matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to:vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N]) device(dev)
  for(int a=0; a<LA; a++) {
    for(int b=0; b<LB; b++) {
      for(int c=0; c<LC; c++) {
        for(int d=0; d<LD; d++) {

          for(int count = 0; count <COUNT; count++) {
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
          }

        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
  gettimeofday(&tv2, NULL);
  long long runtime = (tv2.tv_sec - tv1.tv_sec) * 1000000;
  runtime += tv2.tv_usec - tv1.tv_usec;
  output(DYNAMIC2, dev, runtime);
}

void matrix_vector_collapse2_guided( 
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LA][LB][LC][LD][N],
    int dev)
{
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);
#pragma omp target teams distribute parallel for collapse(2) schedule(guided) \
  map(from:vectorC[dev][0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to:matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to:vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N]) device(dev)
  for(int a=0; a<LA; a++) {
    for(int b=0; b<LB; b++) {
      for(int c=0; c<LC; c++) {
        for(int d=0; d<LD; d++) {

          for(int count = 0; count <COUNT; count++) {
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
          }

        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
  gettimeofday(&tv2, NULL);
  long long runtime = (tv2.tv_sec - tv1.tv_sec) * 1000000;
  runtime += tv2.tv_usec - tv1.tv_usec;
  output(GUIDED2, dev, runtime);
}

void matrix_vector_collapse3_static( 
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LA][LB][LC][LD][N],
    int dev)
{
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);
#pragma omp target teams distribute parallel for collapse(3) schedule(static) \
  map(from:vectorC[dev][0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to:matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to:vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N]) device(dev)
  for(int a=0; a<LA; a++) {
    for(int b=0; b<LB; b++) {
      for(int c=0; c<LC; c++) {
        for(int d=0; d<LD; d++) {

          for(int count = 0; count <COUNT; count++) {
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
          }

        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
  gettimeofday(&tv2, NULL);
  long long runtime = (tv2.tv_sec - tv1.tv_sec) * 1000000;
  runtime += tv2.tv_usec - tv1.tv_usec;
  output(STATIC3, dev, runtime);
}

void matrix_vector_collapse3_dynamic( 
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LA][LB][LC][LD][N],
    int dev)
{
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);
#pragma omp target teams distribute parallel for collapse(3) schedule(dynamic) \
  map(from:vectorC[dev][0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to:matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to:vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N]) device(dev)
  for(int a=0; a<LA; a++) {
    for(int b=0; b<LB; b++) {
      for(int c=0; c<LC; c++) {
        for(int d=0; d<LD; d++) {

          for(int count = 0; count <COUNT; count++) {
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
          }

        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
  gettimeofday(&tv2, NULL);
  long long runtime = (tv2.tv_sec - tv1.tv_sec) * 1000000;
  runtime += tv2.tv_usec - tv1.tv_usec;
  output(DYNAMIC3, dev, runtime);
}

void matrix_vector_collapse3_guided( 
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LA][LB][LC][LD][N],
    int dev)
{
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);
#pragma omp target teams distribute parallel for collapse(3) schedule(guided) \
  map(from:vectorC[dev][0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to:matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to:vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N]) device(dev)
  for(int a=0; a<LA; a++) {
    for(int b=0; b<LB; b++) {
      for(int c=0; c<LC; c++) {
        for(int d=0; d<LD; d++) {

          for(int count = 0; count <COUNT; count++) {
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
          }

        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
  gettimeofday(&tv2, NULL);
  long long runtime = (tv2.tv_sec - tv1.tv_sec) * 1000000;
  runtime += tv2.tv_usec - tv1.tv_usec;
  output(GUIDED3, dev, runtime);
}

void matrix_vector_collapse4_static( 
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LA][LB][LC][LD][N],
    int dev)
{
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);
#pragma omp target teams distribute parallel for collapse(4) schedule(static) \
  map(from:vectorC[dev][0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to:matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to:vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N]) device(dev)
  for(int a=0; a<LA; a++) {
    for(int b=0; b<LB; b++) {
      for(int c=0; c<LC; c++) {
        for(int d=0; d<LD; d++) {

          for(int count = 0; count <COUNT; count++) {
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
          }

        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
  gettimeofday(&tv2, NULL);
  long long runtime = (tv2.tv_sec - tv1.tv_sec) * 1000000;
  runtime += tv2.tv_usec - tv1.tv_usec;
  output(STATIC4, dev, runtime);
}

void matrix_vector_collapse4_dynamic( 
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LA][LB][LC][LD][N],
    int dev)
{
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);
#pragma omp target teams distribute parallel for collapse(4) schedule(dynamic) \
  map(from:vectorC[dev][0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to:matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to:vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N]) device(dev)
  for(int a=0; a<LA; a++) {
    for(int b=0; b<LB; b++) {
      for(int c=0; c<LC; c++) {
        for(int d=0; d<LD; d++) {

          for(int count = 0; count <COUNT; count++) {
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
          }

        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
  gettimeofday(&tv2, NULL);
  long long runtime = (tv2.tv_sec - tv1.tv_sec) * 1000000;
  runtime += tv2.tv_usec - tv1.tv_usec;
  output(DYNAMIC4, dev, runtime);
}

void matrix_vector_collapse4_guided( 
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LA][LB][LC][LD][N],
    int dev)
{
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);
#pragma omp target teams distribute parallel for collapse(4) schedule(guided) \
  map(from:vectorC[dev][0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to:matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to:vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N]) device(dev)
  for(int a=0; a<LA; a++) {
    for(int b=0; b<LB; b++) {
      for(int c=0; c<LC; c++) {
        for(int d=0; d<LD; d++) {

          for(int count = 0; count <COUNT; count++) {
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
          }

        } // end d loop
      } // end c loop
    } // end b loop
  } // end a loop
  gettimeofday(&tv2, NULL);
  long long runtime = (tv2.tv_sec - tv1.tv_sec) * 1000000;
  runtime += tv2.tv_usec - tv1.tv_usec;
  output(GUIDED4, dev, runtime);
}

void kernel(VARIANT var,
            float (*matA)[LB][LC][LD][LE][M][N],
            float (*vecB)[LB][LC][LD][LE][N]) 
{
  float (*vecC)[LA][LB][LC][LD][N] =
    (float (*)[LA][LB][LC][LD][N]) malloc(sizeof(float)*num_dev*LA*LB*LC*LD*N);

#pragma omp parallel for
  for(int device = 0; device < num_dev; device++) {
#ifdef DEBUG
    fprintf(stderr, "Offloading on GPU %d\n", device);
#endif
    switch(var) {
      case STATIC1:
        matrix_vector_collapse1_static(matA, vecB, vecC, device);
        break;
      case STATIC2:
        matrix_vector_collapse2_static(matA, vecB, vecC, device);
        break;
      case STATIC3:
        matrix_vector_collapse3_static(matA, vecB, vecC, device);
        break;
      case STATIC4:
        matrix_vector_collapse4_static(matA, vecB, vecC, device);
        break;
        break;
      case DYNAMIC1:
        matrix_vector_collapse1_dynamic(matA, vecB, vecC, device);
        break;
      case DYNAMIC2:
        matrix_vector_collapse2_dynamic(matA, vecB, vecC, device);
        break;
      case DYNAMIC3:
        matrix_vector_collapse3_dynamic(matA, vecB, vecC, device);
        break;
      case DYNAMIC4:
        matrix_vector_collapse4_dynamic(matA, vecB, vecC, device);
        break;
        break;
      case GUIDED1:
        matrix_vector_collapse1_guided(matA, vecB, vecC, device);
        break;
      case GUIDED2:
        matrix_vector_collapse2_guided(matA, vecB, vecC, device);
        break;
      case GUIDED3:
        matrix_vector_collapse3_guided(matA, vecB, vecC, device);
        break;
      case GUIDED4:
        matrix_vector_collapse4_guided(matA, vecB, vecC, device);
        break;
        break;
      default:
        break;
    }

    print(vecC[device]);
#ifdef VALIDATE
    if(validate(var, vecC[device], device) == 0)
      fprintf(stderr, "PASS matrix_vector_collapse1_static on GPU%d\n", device);
#endif
  }
}

int main() 
{
  printf("Device,mem_to,mem_from,COUNT,LA,LB,LC,LD,LE,M,N,runtime(us),runtime(s)\n");
  mem_to = sizeof(float)*LA*LB*LC*LD*LE*M*N + sizeof(float)*LA*LB*LC*LD*LE*N;
  mem_from = sizeof(float)*LA*LB*LC*LD*N;
#ifdef DEBUG
  fprintf(stderr, "Total memory for matA = %lf\n", sizeof(float)*LA*LB*LC*LD*LE*M*N / 1024.0 / 1024.0 / 1024.0);
  fprintf(stderr, "Total memory for vecB = %lf\n", sizeof(float)*LA*LB*LC*LD*LE*N / 1024.0 / 1024.0 / 1024.0);
  fprintf(stderr, "Total memory for vecC = %lf\n", sizeof(float)*LA*LB*LC*LD*N / 1024.0 / 1024.0 / 1024.0);
  fflush(stderr);
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

  vecCPU = (float (*)[LB][LC][LD][N]) malloc(sizeof(float)*LA*LB*LC*LD*N);

  struct timeval  tv1_cpu, tv2_cpu;
  gettimeofday(&tv1_cpu, NULL);
  matrix_vector_cpu(matA, vecB, vecCPU);
  gettimeofday(&tv2_cpu, NULL);
  long long runtime_cpu = (tv2_cpu.tv_sec - tv1_cpu.tv_sec) * 1000000;
  runtime_cpu += tv2_cpu.tv_usec - tv1_cpu.tv_usec;
  printf("CPU,0,0,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%0.4f\n", COUNT, LA,LB,LC,LD,LE,M,N,runtime_cpu, (double)(runtime_cpu/1000000.0));
  fflush(stdout);

  print(vecCPU);

#ifdef OFF
  num_dev = omp_get_num_devices();
  if(num_dev == 0) {
    fprintf(stderr, "No device available\n");
    exit(-1);
  }
  kernel(STATIC1, matA, vecB);
  kernel(DYNAMIC1, matA, vecB);
  kernel(GUIDED1, matA, vecB);
  kernel(STATIC2, matA, vecB);
  kernel(DYNAMIC2, matA, vecB);
  kernel(GUIDED2, matA, vecB);
  kernel(STATIC3, matA, vecB);
  kernel(DYNAMIC3, matA, vecB);
  kernel(GUIDED3, matA, vecB);
  kernel(STATIC4, matA, vecB);
  kernel(DYNAMIC4, matA, vecB);
  kernel(GUIDED4, matA, vecB);
#endif // End offloading
  return 0;
}
