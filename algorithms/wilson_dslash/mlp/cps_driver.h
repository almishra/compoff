#ifndef _CPS_DRIVER_H
#define _CPS_DRIVER_H

#ifndef LX
#define LX 8
#endif
#ifndef LY
#define LY 16
#endif
#ifndef LZ
#define LZ 16
#endif
#ifndef LT
#define LT 96
#endif
#ifndef LS
#define LS 16
#endif

//#define LX 64
//#define LY 64
//#define LZ 64
//#define LT 96
//#define LS 24

#define MASS 0.33
//#define DSLASH_5_PLUS
//#define GAUGE_COMPRESS
#define PREFETCH_T0(addr,nrOfBytesAhead) _mm_prefetch(((char *)(addr))+nrOfBytesAhead,_MM_HINT_T0)

#define LIONELK_FETCH_DIST 48 // tune this value according to real world experiments

//#define HAND_OPT

#define SPINOR_SIZE 24 
#define HALF_SPINOR_SIZE 12 
#define GAUGE_SIZE 72

#define ABS(x) ((x)<0 ? -(x) : (x))
#define FLOAT
#ifdef FLOAT
typedef float IFloat;
typedef float Float;
#else
typedef double IFloat;
typedef double Float;
#endif

#endif
