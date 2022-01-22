#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string>
#include <typeinfo>

#ifndef N1
#define N1 1000
#endif

#ifndef N2
#define N2 100
#endif

#ifndef TYPE
#define TYPE double
#endif

long mem_to;
long mem_from;
static FILE *fp;

std::string type;

long get_time()
{
  struct timeval  tv;
  gettimeofday(&tv, NULL); 
  return (long)(tv.tv_sec * 1000000 + tv.tv_usec);
}

void add(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
  long start = get_time();
#pragma omp parallel for
  for(int i=0; i<N1; i++)
    for(int j=0; j<N2; j++)
      C[i][j] = A[i][j] + B[i][j];
  long end = get_time();
  fprintf(fp, "add_%s,0,0,%d,%d,%ld\n", type.c_str(), N1, N2, end - start);
}

void add_combine(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute parallel for
  for(int i=0; i<N1; i++)
    for(int j=0; j<N2; j++)
      C[i][j] = A[i][j] + B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "add_combine_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void add_split(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute
  for(int i=0; i<N1; i++)
#pragma omp parallel for
    for(int j=0; j<N2; j++)
      C[i][j] = A[i][j] + B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "add_split_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void add_collapse(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute parallel for collapse(2)
  for(int i=0; i<N1; i++)
    for(int j=0; j<N2; j++)
      C[i][j] = A[i][j] + B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "add_collapse_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void add_combine_swap(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute parallel for
  for(int j=0; j<N2; j++)
    for(int i=0; i<N1; i++)
      C[i][j] = A[i][j] + B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "add_combine_swap_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void add_split_swap(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute
  for(int j=0; j<N2; j++)
#pragma omp parallel for
    for(int i=0; i<N1; i++)
      C[i][j] = A[i][j] + B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "add_split_swap_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void add_collapse_swap(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute parallel for collapse(2)
  for(int j=0; j<N2; j++)
    for(int i=0; i<N1; i++)
      C[i][j] = A[i][j] + B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "add_collapse_swap_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void mult(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
  long start = get_time();
#pragma omp parallel for
  for(int i=0; i<N1; i++)
    for(int j=0; j<N2; j++)
      C[i][j] = A[i][j] * B[i][j];
  long end = get_time();
  fprintf(fp, "mult_%s,0,0,%d,%d,%ld\n", type.c_str(), N1, N2, end - start);
}

void mult_combine(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute parallel for
  for(int i=0; i<N1; i++)
    for(int j=0; j<N2; j++)
      C[i][j] = A[i][j] * B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "mult_combine_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void mult_split(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute
  for(int i=0; i<N1; i++)
#pragma omp parallel for
    for(int j=0; j<N2; j++)
      C[i][j] = A[i][j] * B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "mult_split_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void mult_collapse(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute parallel for collapse(2)
  for(int i=0; i<N1; i++)
    for(int j=0; j<N2; j++)
      C[i][j] = A[i][j] * B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "mult_collapse_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void mult_combine_swap(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute parallel for
  for(int j=0; j<N2; j++)
    for(int i=0; i<N1; i++)
      C[i][j] = A[i][j] * B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "mult_combine_swap_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void mult_split_swap(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute
  for(int j=0; j<N2; j++)
#pragma omp parallel for
    for(int i=0; i<N1; i++)
      C[i][j] = A[i][j] * B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "mult_split_swap_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void mult_collapse_swap(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute parallel for collapse(2)
  for(int j=0; j<N2; j++)
    for(int i=0; i<N1; i++)
      C[i][j] = A[i][j] * B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "mult_collapse_swap_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void div(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
  long start = get_time();
#pragma omp parallel for
  for(int i=0; i<N1; i++)
    for(int j=0; j<N2; j++)
      C[i][j] = A[i][j] / B[i][j];
  long end = get_time();
  fprintf(fp, "div_%s,0,0,%d,%d,%ld\n", type.c_str(), N1, N2, end - start);
}

void div_combine(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute parallel for
  for(int i=0; i<N1; i++)
    for(int j=0; j<N2; j++)
      C[i][j] = A[i][j] / B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "div_combine_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void div_split(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute
  for(int i=0; i<N1; i++)
#pragma omp parallel for
    for(int j=0; j<N2; j++)
      C[i][j] = A[i][j] / B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "div_split_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void div_collapse(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute parallel for collapse(2)
  for(int i=0; i<N1; i++)
    for(int j=0; j<N2; j++)
      C[i][j] = A[i][j] / B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "div_collapse_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void div_combine_swap(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute parallel for
  for(int j=0; j<N2; j++)
    for(int i=0; i<N1; i++)
      C[i][j] = A[i][j] / B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "div_combine_swap_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void div_split_swap(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute
  for(int j=0; j<N2; j++)
#pragma omp parallel for
    for(int i=0; i<N1; i++)
      C[i][j] = A[i][j] / B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "div_split_swap_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
}

void div_collapse_swap(TYPE (*A)[N2], TYPE (*B)[N2], TYPE (*C)[N2])
{
#ifdef MEMCPY
  mem_to = 2*sizeof(TYPE)*N1*N2;
  mem_from = sizeof(TYPE)*N1*N2;
#else
  mem_to = 0;
  mem_from = 0;
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
#pragma omp target teams distribute parallel for collapse(2)
  for(int j=0; j<N2; j++)
    for(int i=0; i<N1; i++)
      C[i][j] = A[i][j] / B[i][j];
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif
  long end = get_time();
  fprintf(fp, "div_collapse_swap_%s,%ld,%ld,%d,%d,%ld\n", type.c_str(), mem_to, mem_from, N1, N2, end - start);
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
  type.assign("double");
  if(typeid(TYPE) == typeid(int))
    type.assign("int");
  else if(typeid(TYPE) == typeid(long))
    type.assign("long");
  else if(typeid(TYPE) == typeid(float))
    type.assign("float");
  fprintf(fp, "Total size for %s = %0.4lf\n", type.c_str(), 3.0*sizeof(TYPE)*N1*N2/1024.0/1024.0/1024.0);

  TYPE (*A)[N2] = (TYPE (*)[N2]) malloc(sizeof(TYPE)*N1*N2);
  TYPE (*B)[N2] = (TYPE (*)[N2]) malloc(sizeof(TYPE)*N1*N2);
  TYPE (*C)[N2] = (TYPE (*)[N2]) malloc(sizeof(TYPE)*N1*N2);

#pragma omp parallel for
  for(int i=0; i<N1; i++)
    for(int j=0; j<N2; j++)
      B[i][j] = 1;

#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])

#ifndef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(alloc: C[0:N1][0:N2])
#endif
  add(A, B, C);
  add_combine(A, B, C);
  add_split(A, B, C);
  add_collapse(A, B, C);
  add_combine_swap(A, B, C);
  add_split_swap(A, B, C);
  add_collapse_swap(A, B, C);

  mult(A, B, C);
  mult_combine(A, B, C);
  mult_split(A, B, C);
  mult_collapse(A, B, C);
  mult_combine_swap(A, B, C);
  mult_split_swap(A, B, C);
  mult_collapse_swap(A, B, C);

  div(A, B, C);
  div_combine(A, B, C);
  div_split(A, B, C);
  div_collapse(A, B, C);
  div_combine_swap(A, B, C);
  div_split_swap(A, B, C);
  div_collapse_swap(A, B, C);

#ifndef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N1][0:N2]) \
  map(from: C[0:N1][0:N2])
#endif

  return 0;
}
