CFLAGS = -Wall -O0 -fopenmp
OFFLOAD_FLAG = -fopenmp-targets=nvptx64-nvidia-cuda
#--cuda-path=$(CUDA_TOOLKIT_ROOT_DIR)
CC=clang++

all: k80 p100 v100 rtx

check-env:
ifndef CUDA_TOOLKIT_ROOT_DIR
    $(error CUDA_TOOLKIT_ROOT_DIR is undefined. Please set CUDA_TOOLKIT_ROOT_DIR to the install location of cuda.)
endif

k80: matrix_vector.cpp
	for A in 1 16 32 48 64 80 96; do \
	for B in 1 16 32 48 64 80 96; do \
	for C in 1 16 32 48 64 80 96; do \
	for D in 1 16 32 48 64 80 96; do \
	for E in 2 4 8; do \
	for COUNT in 10 100 1000; do \
		if [ ! -f output_mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.csv ]; then \
			$(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_37 matrix_vector.cpp -DOFF -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.out; \
			./mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.out; \
		fi \
	done; done; done; done; done; done


p100: matrix_vector.cpp
	for A in 1 16 32 48 64 80 96; do \
	for B in 1 16 32 48 64 80 96; do \
	for C in 1 16 32 48 64 80 96; do \
	for D in 1 16 32 48 64 80 96; do \
	for E in 2 4 8; do \
	for COUNT in 10 100 1000; do \
		if [ ! -f output_mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_p100.csv ]; then \
			$(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_60 matrix_vector.cpp -DOFF -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_p100.out; \
			./mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_p100.out; \
		fi \
	done; done; done; done; done; done


v100: matrix_vector.cpp
	for A in 1 16 32 48 64 80 96; do \
	for B in 1 16 32 48 64 80 96; do \
	for C in 1 16 32 48 64 80 96; do \
	for D in 1 16 32 48 64 80 96; do \
	for E in 2 4 8; do \
	for COUNT in 10 100 1000; do \
		if [ ! -f output_mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.csv ]; then \
			$(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_70 matrix_vector.cpp -DOFF -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.out; \
			./mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.out; \
		fi \
	done; done; done; done; done; done


rtx: matrix_vector.cpp
	for A in 1 16 32 48 64 80 96; do \
	for B in 1 16 32 48 64 80 96; do \
	for C in 1 16 32 48 64 80 96; do \
	for D in 1 16 32 48 64 80 96; do \
	for E in 2 4 8; do \
	for COUNT in 10 100 1000; do \
		if [ ! -f output_mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_rtx.csv ]; then \
			$(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_75 matrix_vector.cpp -DOFF -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_rtx.out; \
			./mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_rtx.out; \
		fi \
	done; done; done; done; done; done


clean:
	$(RM) mv_clang_*.out
