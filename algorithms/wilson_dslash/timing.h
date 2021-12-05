#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>

static double start_walltime;
static unsigned long long start_cycle;

/** Timing function */
double __inline rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  //return (Tp.tv_sec + Tp.tv_usec*1.0e-6);
  return (Tp.tv_sec*1.0e6 + Tp.tv_usec);
}

void __inline init_timer() {
  start_walltime = -1.0;
}

void __inline start_timer() {
  start_walltime = rtclock();
}

double __inline stop_timer() {
  return rtclock() - start_walltime;
}

void __inline reset_timer() {
  start_walltime = -1.0;
}

double __inline get_walltime() {
  return rtclock();
}

double __inline get_start_walltime() {
  return start_walltime;
}

/** Flush the last level cache. Streams a 64MB array into / out of cache. */
void flush_llc() {
  int size_in_mb = 64;
  //printf("Flushing last level cache... ");
  // Seed random # gen
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  srand(Tp.tv_usec);
  
  // Allocate a big array, set first element to random #, traverse it twice
  int i,j;
  double *flush = (double*)malloc(size_in_mb*1024*128*sizeof(double));
  flush[0] = (rand() % 128) * (((double)rand()) / RAND_MAX) + 1;
  for (i = 0; i < 2; i++) {
    for (j = 1; j < size_in_mb*1024*128; j++) {
      flush[j] = flush[j-1]*1.00000000000000001;
    }
  }
  //printf("Finished.\n");
  assert(flush[size_in_mb*1024*128 - 1] != 0.0);
  free(flush);
}

/** Flush the last level cache. Streams a size_in_mb array into / out of cache. */
void flush_any_llc(int size_in_mb) {
  //printf("Flushing last level cache... ");
  // Seed random # gen
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  srand(Tp.tv_usec);
  
  // Allocate a big array, set first element to random #, traverse it twice
  int i,j;
  double *flush = (double*)malloc(size_in_mb*1024*128*sizeof(double));
  flush[0] = (rand() % 128) * (((double)rand()) / RAND_MAX) + 1;
  for (i = 0; i < 2; i++) {
    for (j = 1; j < size_in_mb*1024*128; j++) {
      flush[j] = flush[j-1]*1.00000000000000001;
    }
  }
  //printf("Finished.\n");
  assert(flush[size_in_mb*1024*128 - 1] != 0.0);
  free(flush);
}
