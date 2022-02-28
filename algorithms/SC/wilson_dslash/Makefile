all: k80 p100 v100 rtx

k80: dwf_dslash_4_driver.cpp cps_driver.h timing.h
	$(MAKE) -f Makefile.clang k80
	$(MAKE) -f Makefile.gcc k80

p100: dwf_dslash_4_driver.cpp cps_driver.h timing.h
	$(MAKE) -f Makefile.clang p100
	$(MAKE) -f Makefile.gcc p100

v100: dwf_dslash_4_driver.cpp cps_driver.h timing.h
	$(MAKE) -f Makefile.clang v100
	$(MAKE) -f Makefile.gcc v100
	$(MAKE) -f Makefile.nvc v100

rtx: dwf_dslash_4_driver.cpp cps_driver.h timing.h
	$(MAKE) -f Makefile.clang rtx
	$(MAKE) -f Makefile.gcc rtx
	$(MAKE) -f Makefile.nvc rtx

clean:
	$(RM) dslash_*.out output_*.csv
