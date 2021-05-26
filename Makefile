NVCC = nvcc
CC = g++
CU_FLAGS = -O2 -g --ptxas-options=-v
CFLAGS += -O3 -Wall -g
LFLAGS += -lGL -lGLU -lglut
APPS = demo benchmark accuracy cuda-benchmark

ifdef DATATYPE
CFLAGS += -D DATATYPE=$(DATATYPE)
endif

.PHONY: all

all: $(APPS)

%.o: %.cu
	$(NVCC) $(CU_FLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

demo: demo.c solver.c io.c
	$(CC) $(CFLAGS)  -o $@ $^ $(LFLAGS)

benchmark: benchmark.c solver.c io.c 
	$(CC) $(CFLAGS)  -o $@ $^ $(LFLAGS)

cuda-benchmark: cuda_solver.o solver.o cuda_benchmark.o 
	$(NVCC) $^ -o $@

accuracy: accuracy.c solver.c io.c
	$(CC) $(CFLAGS)  -o $@ $^ $(LFLAGS)

.PHONY: clean
clean:
	rm -f $(APPS) *.o
