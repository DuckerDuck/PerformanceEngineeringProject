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

__global__ void lin_solve_kernel(int N, fluid *x, fluid *x0, float a, float c) 
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j;
	fluid tmp = 0;
	
	// i starts at 1
	i += 1;
	
	if (i <= N) {
		for (j = 1; j <= N; j++) 
		{
			tmp = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)]));
			x[IX(i, j)] = tmp / c;
		}
	}
}

__global__ void set_bnd_cuda(int N, int b, fluid *x)
{
	int i;

	for (i = 1; i <= N; i++)
	{
		x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
		x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
		x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
		x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
	}
	x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
	x[IX(0, N + 1)] = 0.5f * (x[IX(1, N + 1)] + x[IX(0, N)]);
	x[IX(N + 1, 0)] = 0.5f * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
	x[IX(N + 1, N + 1)] = 0.5f * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

void to_device(int N, fluid* a, fluid* b, GPUSTATE gpu)
{
	int size = (N + 2) * (N + 2) * sizeof(fluid);
	
	checkCuda(cudaMemcpy(gpu.a, a, size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(gpu.b, b, size, cudaMemcpyHostToDevice));

}

void to_host(int N, fluid* a, fluid* b, GPUSTATE gpu)
{
	int size = (N + 2) * (N + 2) * sizeof(fluid);
	
	checkCuda(cudaMemcpy(a, gpu.a, size, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(b, gpu.b, size, cudaMemcpyDeviceToHost));

}

void lin_solve_cuda(int N, int b, fluid *x, fluid *x0, float a, float c, GPUSTATE gpu)
{	
	int k;
	int threadBlockSize = 16;

	to_device(N, x, x0, gpu);
	for (k = 0; k < 20; k++)
	{
		lin_solve_kernel<<<N/threadBlockSize + 1, threadBlockSize>>>(N, gpu.a, gpu.b, a, c);
		checkCuda(cudaGetLastError());
		
		set_bnd_cuda<<<1, threadBlockSize>>>(N, b, gpu.a);
		checkCuda(cudaGetLastError());
	}
	to_host(N, x, x0, gpu);
}

void diffuse_cuda(int N, int b, fluid *x, fluid *x0, float diff, float dt, GPUSTATE gpu)
{
	float a = dt * diff * N * N;
	lin_solve_cuda(N, b, x, x0, a, 1 + 4 * a, gpu);
}


void vel_step_cuda(int N, fluid *u, fluid *v, fluid *u0, fluid *v0, float visc, float dt, GPUSTATE gpu)
{
	add_source(N, u, u0, dt);
	add_source(N, v, v0, dt);
	SWAP(u0, u);
	diffuse_cuda(N, 1, u, u0, visc, dt, gpu);
	SWAP(v0, v);
	diffuse_cuda(N, 2, v, v0, visc, dt, gpu);
	project(N, u, v, u0, v0);
	SWAP(u0, u);
	SWAP(v0, v);
	advect(N, 1, u, u0, u0, v0, dt);
	advect(N, 2, v, v0, u0, v0, dt);
	project(N, u, v, u0, v0);
}
