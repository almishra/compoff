# C compiler
CC = g++
ICC = icc
CC_FLAGS = -g -O3 -fopenmp
OFFLOAD_CC_FLAGS = -foffload=nvptx-none

all: needle needle_offload

needle: 
	$(CC) $(CC_FLAGS) needle.cpp -o needle 

needle_offload:
	$(CC) $(CC_FLAGS) $(OFFLOAD_CC_FLAGS) -DOMP_OFFLOAD needle.cpp -o needle_offload

clean:
	rm -f needle needle_offload
