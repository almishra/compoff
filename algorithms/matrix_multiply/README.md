# Matrix Multiplication
This is the matrix multiply algorithm.

To build the program for a particular GPU use the following commands:
```
make <gpu>
```
where gpu could be:
- k80
- p100
- v100
- rtx
This would build the program using all compilers for the given GPU.
<br />

To build the program for a particular set of compiler and GPU use the following commands:
```
make -f Makefile.<compiler> <gpu>
```
where gpu could be from the list mentioned above and compiler could be:
- clang
- gcc
- nvc
<br />

For example, if you want to build with the clang compiler on *Seawulf's K80* node,
run the command:
```
make -f Makefile.clang k80
```
