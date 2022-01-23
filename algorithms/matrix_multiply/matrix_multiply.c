#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <typeinfo>                                                             


#define N 3000
#ifndef TYPE
#define TYPE float
#endif

#ifndef N1
#define N1 1000
#endif

#ifndef N2
#define N2 100
#endif

#ifndef N3
#define N3 1000
#endif

std::string type;
long mem_to;
long mem_from;
static FILE *fp;

long get_time()
{
  struct timeval  tv;
  gettimeofday(&tv, NULL);
  return (long)(tv.tv_sec * 1000000 + tv.tv_usec);
}

void multiply(TYPE (*A)[N2], TYPE (*B)[N3], TYPE (*C)[N3])
{
  long start = get_time();
#pragma omp parallel for
  for(int i=0; i<N1; i++) {
    for(int j=0; j<N2; j++) {
      double sum = 0.0;
      for (int k = 0; k < N3; k++)
        sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
  long end = get_time();
  fprintf(fp, "matrix_mult_%s,0,0,%d,%d,%d,%ld\n", type.c_str(), N1, N2, N3, end - start);
}

void multiply_combine(TYPE (*A)[N2], TYPE (*B)[N3], TYPE (*C)[N3])
{
#ifdef MEMCPY
  mem_to = sizeof(TYPE)*N1*N2 + sizeof(TYPE)*N2*N3;
  mem_from = sizeof(TYPE)*N1*N3;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(alloc: C[0:N1][0:N3])
#endif
#pragma omp target teams distribute parallel for
  for(int i=0; i<N1; i++) {
    for(int j=0; j<N2; j++) {
      double sum = 0.0;
      for (int k = 0; k < N3; k++)
        sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(from: C[0:N1][0:N3])
#endif
  long end = get_time();
  fprintf(fp, "matrix_mult_combine_%s,0,0,%d,%d,%d,%ld\n", type.c_str(), N1, N2, N3, end - start);
}

void multiply_split(TYPE (*A)[N2], TYPE (*B)[N3], TYPE (*C)[N3])
{
#ifdef MEMCPY
  mem_to = sizeof(TYPE)*N1*N2 + sizeof(TYPE)*N2*N3;
  mem_from = sizeof(TYPE)*N1*N3;
#else   
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(alloc: C[0:N1][0:N3])
#endif
#pragma omp target teams distribute
  for(int i=0; i<N1; i++) {
#pragma omp parallel for
    for(int j=0; j<N2; j++) {
      double sum = 0.0;
      for (int k = 0; k < N3; k++)
        sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(from: C[0:N1][0:N3])
#endif
  long end = get_time();
  fprintf(fp, "matrix_mult_split_%s,0,0,%d,%d,%d,%ld\n", type.c_str(), N1, N2, N3, end - start);
}

void multiply_collapse(TYPE (*A)[N2], TYPE (*B)[N3], TYPE (*C)[N3])
{
#ifdef MEMCPY
  mem_to = sizeof(TYPE)*N1*N2 + sizeof(TYPE)*N2*N3;
  mem_from = sizeof(TYPE)*N1*N3;
#else   
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(alloc: C[0:N1][0:N3])
#endif
#pragma omp target teams distribute parallel for collapse(2)
  for(int i=0; i<N1; i++) {
    for(int j=0; j<N2; j++) {
      double sum = 0.0;
      for (int k = 0; k < N3; k++)
        sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(from: C[0:N1][0:N3])
#endif
  long end = get_time();
  fprintf(fp, "matrix_mult_collapse_%s,0,0,%d,%d,%d,%ld\n", type.c_str(), N1, N2, N3, end - start);
}

void multiply_combine_swap(TYPE (*A)[N2], TYPE (*B)[N3], TYPE (*C)[N3]) 
{
#ifdef MEMCPY
  mem_to = sizeof(TYPE)*N1*N2 + sizeof(TYPE)*N2*N3;
  mem_from = sizeof(TYPE)*N1*N3;
#else   
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(alloc: C[0:N1][0:N3])
#endif
#pragma omp target teams distribute parallel for
  for(int j=0; j<N2; j++) {
    for(int i=0; i<N1; i++) {
      double sum = 0.0;
      for (int k = 0; k < N3; k++)
        sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(from: C[0:N1][0:N3])
#endif
  long end = get_time();
  fprintf(fp, "matrix_mult_combine_swap_%s,0,0,%d,%d,%d,%ld\n", type.c_str(), N1, N2, N3, end - start);
}

void multiply_split_swap(TYPE (*A)[N2], TYPE (*B)[N3], TYPE (*C)[N3])
{
#ifdef MEMCPY
  mem_to = sizeof(TYPE)*N1*N2 + sizeof(TYPE)*N2*N3;
  mem_from = sizeof(TYPE)*N1*N3;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(alloc: C[0:N1][0:N3])
#endif
#pragma omp target teams distribute
  for(int j=0; j<N2; j++) {
#pragma omp parallel for
    for(int i=0; i<N1; i++) {
      double sum = 0.0;
      for (int k = 0; k < N3; k++)
        sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(from: C[0:N1][0:N3])
#endif
  long end = get_time();
  fprintf(fp, "matrix_mult_split_swap_%s,0,0,%d,%d,%d,%ld\n", type.c_str(), N1, N2, N3, end - start);
}

void multiply_collapse_swap(TYPE (*A)[N2], TYPE (*B)[N3], TYPE (*C)[N3])
{
#ifdef MEMCPY
  mem_to = sizeof(TYPE)*N1*N2 + sizeof(TYPE)*N2*N3;
  mem_from = sizeof(TYPE)*N1*N3;
#else   
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(alloc: C[0:N1][0:N3])
#endif
#pragma omp target teams distribute parallel for collapse(2)
  for(int j=0; j<N2; j++) {
    for(int i=0; i<N1; i++) {
      double sum = 0.0;
      for (int k = 0; k < N3; k++)
        sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(from: C[0:N1][0:N3])
#endif
  long end = get_time();
  fprintf(fp, "matrix_mult_collapse_%s,0,0,%d,%d,%d,%ld\n", type.c_str(), N1, N2, N3, end - start);
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
  type.assign("float");
  if(typeid(TYPE) == typeid(int))
    type.assign("int");
  else if(typeid(TYPE) == typeid(long))
    type.assign("long");
  else if(typeid(TYPE) == typeid(float))
    type.assign("float");
  fprintf(fp, "Total size for %s = %0.4lf\n", type.c_str(), 3.0*sizeof(TYPE)*N1*N2/1024.0/1024.0/1024.0);

  TYPE (*A)[N2] = (TYPE (*)[N2]) malloc(sizeof(TYPE)*N1*N2);
  TYPE (*B)[N3] = (TYPE (*)[N3]) malloc(sizeof(TYPE)*N2*N3);
  TYPE (*C)[N3] = (TYPE (*)[N3]) malloc(sizeof(TYPE)*N1*N3);

#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(alloc: C[0:N1][0:N3])
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(from: C[0:N1][0:N3])

#ifndef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(alloc: C[0:N1][0:N3])
#endif
  multiply(A, B, C);
  multiply_combine(A, B, C);
  multiply_split(A, B, C);
  multiply_collapse(A, B, C);
  multiply_combine_swap(A, B, C);
  multiply_split_swap(A, B, C);
  multiply_collapse_swap(A, B, C);
#ifndef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(from: C[0:N1][0:N3])
#endif

  return 0;
}

