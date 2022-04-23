clang -c -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_75 -I../common lud.c -o lud.o -DOMP_OFFLOAD
clang -c -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_75 -I../common lud_omp.c -o lud_omp.o -DOMP_OFFLOAD
clang -c -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_75 -I../common ../common/common.c -o ../common/common.o -DOMP_OFFLOAD
clang -fopenmp -fopenmp-targets=nvptx64 -L/lustre/projects/compoff-group/opt/llvm/lib -Xopenmp-target -march=sm_75 -o lud_omp lud.o lud_omp.o ../common/common.o -lm  -DOMP_OFFLOAD
