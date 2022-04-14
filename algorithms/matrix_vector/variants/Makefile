#CFLAGS = -Wall -O3 -fopenmp -DMEMCPY
CFLAGS = -Wall -O3 -fopenmp
OFFLOAD_FLAG = -fopenmp-targets=nvptx64
CC=clang++

all: k80 p100 v100 rtx

k80: matrix_vector.cpp
	for A in 96 64 32; do \
	for B in 96 64 32; do \
	for C in 96 64 32; do \
	for D in 96 64 32; do \
	for E in 2 4 8; do \
	for COUNT in 10 100 500; do \
		if [ ! -f output_mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.csv ]; then \
			num=1; \
			while [[ $$num -lt 10 ]] ; do \
				$(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_70 -c __GEN_matrix_vector_VARIANT_0$${num}.cpp -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o __GEN_matrix_vector_VARIANT_0$${num}_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o; \
				((num = num + 1)); \
			done; \
			while [[ $$num -le 42 ]] ; do \
				$(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_70 -c __GEN_matrix_vector_VARIANT_$${num}.cpp -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o __GEN_matrix_vector_VARIANT_$${num}_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o; \
				((num = num + 1)); \
			done; \
      $(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_70 __GEN_matrix_vector_VARIANT_01_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_02_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_03_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_04_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_05_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_06_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_07_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_08_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_09_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_10_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_11_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_12_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_13_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_14_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_15_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_16_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_17_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_18_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_19_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_20_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_21_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_22_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_23_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_24_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_25_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_26_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_27_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_28_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_29_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_30_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_31_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_32_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_33_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_34_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_35_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_36_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_37_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_38_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_39_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_40_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_41_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_42_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o matrix_vector.cpp -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.out; \
			./mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.out; \
		fi; \
		if [ ! -f output_mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80_memcpy.csv ]; then \
			num2=43; num=1; \
			while [[ $$num -lt 10 ]] ; do \
				$(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_70 -DMEMCPY -c __GEN_matrix_vector_VARIANT_0$${num}.cpp -o -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o __GEN_matrix_vector_VARIANT_$${num2}_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o; \
				((num = num + 1)); \
				((num2 = num2 + 1)); \
			done; \
			while [[ $$num -le 42 ]] ; do \
				$(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_70 -DMEMCPY -c __GEN_matrix_vector_VARIANT_$${num}.cpp -o -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o __GEN_matrix_vector_VARIANT_$${num2}_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o; \
				((num = num + 1)); \
				((num2 = num2 + 1)); \
			done; \
      $(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_70 -DMEMCPY __GEN_matrix_vector_VARIANT_43_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_44_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_45_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_46_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_47_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_48_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_49_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_50_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_51_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_52_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_53_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_54_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_55_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_56_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_57_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_58_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_59_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_60_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_61_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_62_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_63_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_64_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_65_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_66_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_67_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_68_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_69_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_70_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_71_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_72_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_73_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_74_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_75_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_76_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_77_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_78_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_79_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_80_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_81_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_82_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_83_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o __GEN_matrix_vector_VARIANT_84_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80.o matrix_vector.cpp -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80_memcpy.out; \
			./mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_k80_memcpy.out; \
		fi \
	done; done; done; done; done; done


v100: matrix_vector.cpp
	for A in 96 64 32; do \
	for B in 96 64 32; do \
	for C in 96 64 32; do \
	for D in 96 64 32; do \
	for E in 2 4 8; do \
	for COUNT in 10 100 500; do \
		if [ ! -f output_mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.csv ]; then \
			num=1; \
			while [[ $$num -lt 10 ]] ; do \
				$(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_70 -c __GEN_matrix_vector_VARIANT_0$${num}.cpp -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o __GEN_matrix_vector_VARIANT_0$${num}_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o; \
				((num = num + 1)); \
			done; \
			while [[ $$num -le 42 ]] ; do \
				$(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_70 -c __GEN_matrix_vector_VARIANT_$${num}.cpp -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o __GEN_matrix_vector_VARIANT_$${num}_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o; \
				((num = num + 1)); \
			done; \
      $(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_70 __GEN_matrix_vector_VARIANT_01_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_02_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_03_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_04_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_05_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_06_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_07_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_08_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_09_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_10_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_11_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_12_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_13_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_14_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_15_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_16_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_17_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_18_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_19_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_20_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_21_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_22_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_23_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_24_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_25_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_26_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_27_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_28_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_29_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_30_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_31_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_32_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_33_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_34_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_35_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_36_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_37_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_38_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_39_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_40_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_41_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_42_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o matrix_vector.cpp -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.out; \
			./mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.out; \
		fi; \
		if [ ! -f output_mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100_memcpy.csv ]; then \
			num2=43; num=1; \
			while [[ $$num -lt 10 ]] ; do \
				$(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_70 -DMEMCPY -c __GEN_matrix_vector_VARIANT_0$${num}.cpp -o -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o __GEN_matrix_vector_VARIANT_$${num2}_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o; \
				((num = num + 1)); \
				((num2 = num2 + 1)); \
			done; \
			while [[ $$num -le 42 ]] ; do \
				$(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_70 -DMEMCPY -c __GEN_matrix_vector_VARIANT_$${num}.cpp -o -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o __GEN_matrix_vector_VARIANT_$${num2}_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o; \
				((num = num + 1)); \
				((num2 = num2 + 1)); \
			done; \
      $(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_70 -DMEMCPY __GEN_matrix_vector_VARIANT_43_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_44_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_45_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_46_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_47_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_48_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_49_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_50_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_51_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_52_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_53_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_54_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_55_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_56_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_57_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_58_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_59_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_60_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_61_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_62_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_63_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_64_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_65_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_66_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_67_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_68_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_69_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_70_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_71_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_72_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_73_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_74_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_75_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_76_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_77_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_78_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_79_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_80_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_81_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_82_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_83_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o __GEN_matrix_vector_VARIANT_84_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100.o matrix_vector.cpp -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100_memcpy.out; \
			./mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_v100_memcpy.out; \
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
			$(CC) $(CFLAGS) $(OFFLOAD_FLAG) -Xopenmp-target -march=sm_75 matrix_vector.cpp -DLA=$${A} -DLB=$${B} -DLC=$${C} -DLD=$${D} -DLE=$${E} -DCOUNT=$${COUNT} -o mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_rtx.out; \
			./mv_clang_$${A}_$${B}_$${C}_$${D}_$${E}_$${COUNT}_rtx.out; \
		fi \
	done; done; done; done; done; done


clean:
	$(RM) mv_clang_*.out __GEN*.o
