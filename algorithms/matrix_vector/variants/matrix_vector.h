#ifndef __MATRIX_VECTOR_H__
#define __MATRIX_VECTOR_H__

#include <stdlib.h>
#include <stdio.h>
#include <string>
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
#ifndef COUNT
#define COUNT 1000
#endif

static FILE *fp;
static float (*vecCPU)[LB][LC][LD][N];
static float (*vecGPU)[LB][LC][LD][N];
static long mem_to;
static long mem_from;
static long mem_alloc;
static long mem_delete;

void matrix_vector(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);

void matrix_vector_variant1(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant2(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant3(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant4(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant5(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant6(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant7(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant8(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant9(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant10(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant11(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant12(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant13(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant14(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant15(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant16(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant17(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant18(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant19(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant20(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant21(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant22(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant23(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant24(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant25(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant26(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant27(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant28(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant29(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant30(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant31(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant32(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant33(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant34(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant35(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant36(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant37(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant38(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant39(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant40(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant41(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);
void matrix_vector_variant42(float (*matrixA)[LB][LC][LD][LE][M][N], float (*vectorB)[LB][LC][LD][LE][N], float (*vectorC)[LB][LC][LD][N]);

#endif //End __MATRIX_VECTOR_H__ 
