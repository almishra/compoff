#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <omp.h>
#include <string>

#ifndef N1
#define N1 1000
#endif

#ifndef N2
#define N2 100
#endif

#ifndef N3
#define N3 1000
#endif

static FILE *fp;

long get_time()
{
  struct timeval  tv;
  gettimeofday(&tv, NULL);
  return (long)(tv.tv_sec * 1000000 + tv.tv_usec);
}

void multiply_combine(float (*A)[N2], float (*B)[N3], float (*C)[N3])
{
  long start = get_time();
#pragma omp target teams distribute parallel for
  for(int i=0; i<N1; i++) {
    for(int j=0; j<N3; j++) {
      double sum = 0.0;
      for (int k = 0; k < N2; k++)
        sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
  long end = get_time();
  fprintf(fp, "matrix_multiply_combine,%d,%d,%d,%ld\n", N1, N2, N3, end - start);
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

  float (*A)[N2] = (float (*)[N2]) malloc(sizeof(float)*N1*N2);
  float (*B)[N3] = (float (*)[N3]) malloc(sizeof(float)*N2*N3);
  float (*C)[N3] = (float (*)[N3]) malloc(sizeof(float)*N1*N3);

#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(alloc: C[0:N1][0:N3])
  multiply_combine(A, B, C);
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(from: C[0:N1][0:N3])

  return 0;
}

