NVCC = nvcc
CC = g++
CU_FLAGS = -O3 -g --ptxas-options=-v --gpu-architecture=sm_53
CFLAGS += -O3 -Wall -g
LFLAGS += -lGL -lGLU -lglut
APPS = demo benchmark accuracy cuda-benchmark cuda-demo

ifdef DATATYPE
CFLAGS += -D DATATYPE=$(DATATYPE)
endif

ifdef HALF_PRECISION
CU_FLAGS += -D HALF_PRECISION
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

cuda-demo: cuda_solver.o solver.o cuda_demo.o io.o
	$(NVCC) $^ -o $@ $(LFLAGS)

accuracy: accuracy.c solver.c io.c
	$(CC) $(CFLAGS)  -o $@ $^ $(LFLAGS)

.PHONY: clean
clean:
	rm -f $(APPS) *.o
