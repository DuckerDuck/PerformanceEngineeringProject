#include <stdlib.h>
#include <stdio.h>

#include "solver.h"
#include "cuda_solver.h"


void checkCuda(cudaError_t result) 
{
	if (result != cudaSuccess) {
	   printf("CUDA call failed.\n");
	   exit(1);
	}
 }

__global__ void lin_solve_kernel(int N, int b, fluid *x, fluid *x0, float a, float c) 
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	// tmp = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)]));
	// x[IX(i, j)] = tmp / c;
}

void lin_solve_cuda(int N, int b, fluid *x, fluid *x0, float a, float c, GPUSTATE gpu)
{
	
	int i, j, k;
	fluid tmp = 0;
	
	int threadBlockSize = 512;
	checkCuda(cudaGetLastError());

	for (k = 0; k < 20; k++)
	{
		FOR_EACH_CELL
		tmp = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)]));
		x[IX(i, j)] = tmp / c;
		END_FOR
		set_bnd(N, b, x);
	}
}