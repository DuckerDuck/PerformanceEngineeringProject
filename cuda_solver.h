#ifndef _CUDA_SOLVER_H
#define _CUDA_SOLVER_H
#include "solver.h"

typedef struct GPUSTATE {
	fluid *a, *b;
} GPUSTATE;

void lin_solve_cuda(int N, int b, fluid *x, fluid *x0, float a, float c, GPUSTATE gpu);
void checkCuda(cudaError_t result);
#endif