CC=g++
NCC=nvcc
CFLAGS=-c
OFLAGS=-o
LIBPATHFLAG=-L/usr/local/cuda/lib64
RTFLAG=-lcudart

runtest: test.o utils.o
	$(CC) $(OFLAGS) runtest test.o utils.o $(LIBPATHFLAG) $(RTFLAG)

utils.o: utils.cu
	$(NCC) $(CFLAGS) utils.cu

test.o: test.c
	$(CC) $(CFLAGS) test.c

clean: *.o
	rm *.o