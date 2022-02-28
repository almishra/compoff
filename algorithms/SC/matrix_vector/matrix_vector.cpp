#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <omp.h>
#include <sys/time.h>
#include <unistd.h>

#ifndef LA
#define LA 10
#endif
#ifndef LB
#define LB 10
#endif
#ifndef LC
#define LC 10
#endif
#ifndef LD
#define LD 10
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
#define COUNT 10
#endif

static FILE *fp;
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
          fprintf(stderr, "\n");
        }
  fflush(stderr);
#endif
}

int validate(float (*vecC)[LB][LC][LD][N])
{
  for(int a=0; a<LA; a++)
    for(int b=0; b<LB; b++)
      for(int c=0; c<LC; c++)
        for(int d=0; d<LD; d++)
          for(int n=0; n<N; n++)
            if(vecCPU[a][b][c][d][n] != vecC[a][b][c][d][n]) {
              fprintf(stderr, "FAILED on (%d,%d,%d,%d,%d)\n", a, b, c, d, n);
              return -1;
            }
  fprintf(stderr, "Validation Successful\n");
  return 0;
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

void matrix_vector_gpu(
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LB][LC][LD][N])
{
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);
#ifdef MEMCPY
#pragma omp target enter data \
  map(alloc:vectorC[0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to:matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to:vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N])
#endif
#pragma omp target teams distribute parallel for
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
#ifdef MEMCPY
#pragma omp target exit data \
  map(from:vectorC[0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(delete:matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(delete:vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N])
#endif
  gettimeofday(&tv2, NULL);
  long long runtime = (tv2.tv_sec - tv1.tv_sec) * 1000000;
  runtime += tv2.tv_usec - tv1.tv_usec;
  fprintf(fp, "GPU,base,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%0.4f\n",
          mem_to, mem_from, COUNT, LA, LB, LC, LD, LE, M, N, runtime,
          (double)(runtime/1000000.0));
  fflush(fp);
}

int main(int argc, char **argv)
{
  std::string output_file_name;
  if(argc > 1) {
    output_file_name = argv[1];
  } else {
    output_file_name = argv[0];
    output_file_name = output_file_name.substr(output_file_name.find_last_of("/\\")+1);
    output_file_name = output_file_name.substr(0, output_file_name.size() - 3);
    output_file_name = "output_" + output_file_name + "csv";
  }
  fprintf(stderr, "%s\n", output_file_name.c_str());
  fp = fopen(output_file_name.c_str(), "w");

  fprintf(fp, "Device,scheduling,collapse,mem_to,mem_from,COUNT,LA,LB,LC,LD,LE,M,N,runtime(us),runtime(s)\n");
#ifdef DEBUG
  fprintf(stderr, "Total memory for matA = %lf\n", sizeof(float)*LA*LB*LC*LD*LE*M*N / 1024.0 / 1024.0 / 1024.0);
  fprintf(stderr, "Total memory for vecB = %lf\n", sizeof(float)*LA*LB*LC*LD*LE*N / 1024.0 / 1024.0 / 1024.0);
  fprintf(stderr, "Total memory for vecGPU = %lf\n", sizeof(float)*LA*LB*LC*LD*N / 1024.0 / 1024.0 / 1024.0);
  fflush(stderr);
#endif
  float (*matA)[LB][LC][LD][LE][M][N] =
    (float (*)[LB][LC][LD][LE][M][N]) malloc(sizeof(float)*LA*LB*LC*LD*LE*M*N);
  float (*vecB)[LB][LC][LD][LE][N] =
    (float (*)[LB][LC][LD][LE][N]) malloc(sizeof(float)*LA*LB*LC*LD*LE*N);
  float (*vecGPU)[LB][LC][LD][N] = 
    (float (*)[LB][LC][LD][N]) malloc(sizeof(float)*LA*LB*LC*LD*N);

#pragma omp target map(matA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(vecB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N]) \
  map(alloc: vecGPU[0:LA][0:LB][0:LC][0:LD][0:N])
  {}

  for(int a=0; a<LA; a++)
    for(int b=0; b<LB; b++)
      for(int c=0; c<LC; c++)
        for(int d=0; d<LD; d++)
          for(int e=0;e<LE;e++)
            for(int m=0;m<M;m++)
              for(int n=0;n<N;n++)
                  matA[a][b][c][d][e][m][n] = 0.1;

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
  fprintf(fp, "CPU,0,0,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%0.4f\n", COUNT, LA,LB,LC,LD,LE,M,N,runtime_cpu, (double)(runtime_cpu/1000000.0));
  fflush(fp);

  print(vecCPU);

#ifdef OFF
  num_dev = omp_get_num_devices();
  if(num_dev == 0) {
    fprintf(stderr, "No device available\n");
    exit(-1);
  }
#ifndef MEMCPY
  mem_to = 0;
  mem_from = 0;
#pragma omp target enter data map(to: matA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to: vecB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N]) \
  map(alloc: vecGPU[0:LA][0:LB][0:LC][0:LD][0:N])
#else
  mem_to = sizeof(float)*LA*LB*LC*LD*LE*M*N + sizeof(float)*LA*LB*LC*LD*LE*N;
  mem_from = sizeof(float)*LA*LB*LC*LD*N;
#endif
  matrix_vector_gpu(matA, vecB, vecGPU);
#ifndef MEMCPY
#pragma omp target exit data map(delete: matA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(delete: vecB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N]) \
  map(from: vecGPU[0:LA][0:LB][0:LC][0:LD][0:N])
#endif

#ifdef DEBUG
  validate(vecGPU);
#endif

#endif // End offloading

  return 0;
}
