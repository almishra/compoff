all: k80 p100 v100 rtx

k80: matrix_vector.cpp
	$(MAKE) -f Makefile.clang k80
	$(MAKE) -f Makefile.gcc k80

p100: matrix_vector.cpp
	$(MAKE) -f Makefile.clang p100
	$(MAKE) -f Makefile.gcc p100

v100: matrix_vector.cpp
	$(MAKE) -f Makefile.clang v100
	$(MAKE) -f Makefile.gcc v100
	$(MAKE) -f Makefile.nvc v100

rtx: matrix_vector.cpp
	$(MAKE) -f Makefile.clang rtx
	$(MAKE) -f Makefile.gcc rtx
	$(MAKE) -f Makefile.nvc rtx

clean:
	$(RM) mv_*.out output_*.csv
