/*
 * =====================================================================================
 *
 *       Filename:  suite.c
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"

#ifndef MATRIX_DIM
#define MATRIX_DIM 100
#endif

static int do_verify = 0;
int omp_num_threads = 40;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {0,0,0,0}
};

extern void
lud_omp(float *m, int matrix_dim);


int
main ( int argc, char *argv[] )
{
  int matrix_dim = 100; /* default size */
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULL;
  float *m, *mm;
  int blockSize;
    //printf("BlockSize and matrix dim are %d, %d\n", blockSize, matrix_dim);
    // ret = create_matrix(&m, MATRIX_DIM);
    // if (ret != RET_SUCCESS) {
    //   m = NULL;
    //   fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
    //   exit(EXIT_FAILURE);
    // }
    // lud_omp(m, MATRIX_DIM);
  //printf("matrix_dim,block_size,mem_to,mem_from,runtime(s),runtime(us)\n");
  for(;matrix_dim<=10000;){
    ret = create_matrix(&m, matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
      exit(EXIT_FAILURE);
    }
    lud_omp(m, matrix_dim);
    free(m);
  }

  return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */
