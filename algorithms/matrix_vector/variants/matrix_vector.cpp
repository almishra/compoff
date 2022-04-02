#include "matrix_vector.h"

long get_time()
{
  struct timeval  tv;
  gettimeofday(&tv, NULL);
  return (long)(tv.tv_sec * 1000000 + tv.tv_usec);
}

void matrix_vector( 
    float (*matrixA)[LB][LC][LD][LE][M][N],
    float (*vectorB)[LB][LC][LD][LE][N], 
    float (*vectorC)[LB][LC][LD][N])
{
  omp_set_num_threads(28);
#ifdef MEMCPY
#pragma omp target enter data \
  map(alloc: vectorC[0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to: matrixA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to: vectorB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N])
#endif
#pragma omp target
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
  printf("%s\n", output_file_name.c_str());
  fp = fopen(output_file_name.c_str(), "w");

#ifdef MEMCPY
  mem_to = sizeof(float)*LA*LB*LC*LD*LE*M*N + sizeof(float)*LA*LB*LC*LD*LE*N;
  mem_from = sizeof(float)*LA*LB*LC*LD*N;
  mem_alloc = sizeof(float)*LA*LB*LC*LD*N;
  mem_delete = sizeof(float)*LA*LB*LC*LD*LE*M*N + sizeof(float)*LA*LB*LC*LD*LE*N;
#else
  mem_to = 0;
  mem_from = 0;
  mem_alloc = 0;
  mem_delete = 0;
#endif

#ifdef DEBUG
  fprintf(stderr, "Total memory for matA = %lf\n", sizeof(float)*LA*LB*LC*LD*LE*M*N / 1024.0 / 1024.0 / 1024.0);
  fprintf(stderr, "Total memory for vecB = %lf\n", sizeof(float)*LA*LB*LC*LD*LE*N / 1024.0 / 1024.0 / 1024.0);
  fprintf(stderr, "Total memory for vecC = %lf\n", sizeof(float)*LA*LB*LC*LD*N / 1024.0 / 1024.0 / 1024.0);
  fflush(stderr);
#endif
  float (*matA)[LB][LC][LD][LE][M][N] =
    (float (*)[LB][LC][LD][LE][M][N]) malloc(sizeof(float)*LA*LB*LC*LD*LE*M*N);
  float (*vecB)[LB][LC][LD][LE][N] =
    (float (*)[LB][LC][LD][LE][N]) malloc(sizeof(float)*LA*LB*LC*LD*LE*N);

#pragma omp target map(matA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(vecB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N])
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
                vecB[a][b][c][d][e][n] = 0.5;
  long start;
  long end;
  long runtime;
/*
  vecCPU = (float (*)[LB][LC][LD][N]) malloc(sizeof(float)*LA*LB*LC*LD*N);

#ifndef MEMCPY
#pragma omp target enter data \
  map(alloc: vecCPU[0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to: matA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to: vecB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N])
#endif
  long start = get_time();
  matrix_vector(matA, vecB, vecCPU);
  long end = get_time();
  long runtime = end - start;
  fprintf(fp, "Variant,Distribute Collapse,Parallel Position,Collapse,Schedule,mem_to,mem_from,mem_alloc,mem_delete,COUNT,LA,LB,LC,LD,LE,M,N,runtime(us),runtime(s)\n");
  fprintf(fp, "GPU,NULL,NULL,NULL,NULL,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n", mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC,
               LD, LE, M, N, runtime, (double)(runtime/1000000.0));
  fflush(fp);
#ifndef MEMCPY
#pragma omp target exit data \
  map(from: vecCPU[0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(delete: matA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(delete: vecB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N])
#endif
*/

  vecGPU = (float (*)[LB][LC][LD][N]) malloc(sizeof(float)*LA*LB*LC*LD*N);

#ifndef MEMCPY
#pragma omp target enter data \
  map(alloc: vecGPU[0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(to: matA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(to: vecB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N])
#endif

  start = get_time();
  matrix_vector_variant1(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_43,1,1,4,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "﻿Variant_1,1,1,4,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant2(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_44,1,1,4,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_2,1,1,4,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant3(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_45,1,1,4,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_3,1,1,4,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant4(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_46,1,1,3,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_4,1,1,3,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant5(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_47,1,1,3,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_5,1,1,3,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant6(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_48,1,1,3,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_6,1,1,3,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant7(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_49,1,1,2,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_7,1,1,2,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant8(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_50,1,1,2,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_8,1,1,2,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant9(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_51,1,1,2,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_9,1,1,2,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant10(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_52,1,1,1,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_10,1,1,1,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant11(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_53,1,1,1,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_11,1,1,1,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant12(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_54,1,1,1,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_12,1,1,1,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant13(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_55,1,2,3,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_13,1,2,3,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant14(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_56,1,2,3,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_14,1,2,3,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant15(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_57,1,2,3,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_15,1,2,3,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant16(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_58,1,2,2,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_16,1,2,2,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant17(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_59,1,2,2,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_17,1,2,2,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant18(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_60,1,2,2,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_18,1,2,2,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant19(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_61,1,2,1,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_19,1,2,1,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant20(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_62,1,2,1,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_20,1,2,1,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant21(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_63,1,2,1,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_21,1,2,1,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant22(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_64,1,3,2,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_22,1,3,2,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant23(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_65,1,3,2,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_23,1,3,2,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant24(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_66,1,3,2,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_24,1,3,2,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant25(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_67,1,3,1,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_25,1,3,1,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant26(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_68,1,3,1,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_26,1,3,1,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant27(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_69,1,3,1,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_27,1,3,1,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant28(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_70,1,4,1,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_28,1,4,1,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant29(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_71,1,4,1,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_29,1,4,1,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant30(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_72,1,4,1,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_30,1,4,1,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant31(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_73,2,3,2,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_31,2,3,2,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant32(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_74,2,3,2,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_32,2,3,2,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant33(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_75,2,3,2,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_33,2,3,2,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant34(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_76,2,3,1,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_34,2,3,1,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant35(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_77,2,3,1,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_35,2,3,1,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant36(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_78,2,3,1,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_36,2,3,1,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant37(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_79,2,4,1,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_37,2,4,1,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant38(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_80,2,4,1,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_38,2,4,1,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant39(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_81,2,4,1,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_39,2,4,1,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant40(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_82,3,4,1,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_40,3,4,1,Static,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant41(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_83,3,4,1,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_41,3,4,1,Dynamic,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

  start = get_time();
  matrix_vector_variant42(matA, vecB, vecGPU);
  end = get_time();
  runtime = end - start;
#ifdef MEMCPY
  fprintf(fp, "variant_84,3,4,1,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#else
  fprintf(fp, "Variant_42,3,4,1,Guided,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%0.6f\n",
               mem_to, mem_from, mem_alloc, mem_delete, COUNT, LA, LB, LC, LD, LE,
               M, N, runtime, (double)(runtime/1000000.0));
#endif
  fflush(fp);

#ifndef MEMCPY
#pragma omp target exit data \
  map(from: vecGPU[0:LA][0:LB][0:LC][0:LD][0:N]) \
  map(delete: matA[0:LA][0:LB][0:LC][0:LD][0:LE][0:M][0:N]) \
  map(delete: vecB[0:LA][0:LB][0:LC][0:LD][0:LE][0:N])
#endif

  return 0;
}
