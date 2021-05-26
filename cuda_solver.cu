#include <stdlib.h>
#include <stdio.h>
#include <cuda_fp16.h>

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

__global__ void lin_solve_kernel_half(int N, fluid *x, fluid *x0, float a, float c) 
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j;
	
	half ha = __float2half(a);
	half ca = __float2half(c);
	half tmp, k, l, m, o, p;
	
	// i starts at 1
	i += 1;

	if (i <= N) {
		for (j = 1; j <= N; j++) 
		{
			k = __float2half(x0[IX(i, j)]);
			l = __float2half(x[IX(i - 1, j)]);
			m = __float2half(x[IX(i + 1, j)]);
			o = __float2half(x[IX(i, j - 1)]);
			p = __float2half(x[IX(i, j + 1)]);
			// tmp = k + ha * (l + m + o + p);
			tmp = __hfma(ha, __hadd(__hadd(__hadd(l, m), o), p), k);
			
			x[IX(i, j)] = __half2float(tmp / ca);
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
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	int size = (N + 2) * (N + 2) * sizeof(fluid);
	
	checkCuda(cudaMemcpy(gpu.a, a, size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(gpu.b, b, size, cudaMemcpyHostToDevice));

	cudaEventRecord(stop, 0);
	float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("to_device: %f ms\n", elapsedTime);
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

	to_device(N, x, x0, gpu);
	for (k = 0; k < 20; k++)
	{
		LINSOLVE_KERNEL<<<N/BLOCKSIZE + 1, BLOCKSIZE>>>(N, gpu.a, gpu.b, a, c);
		checkCuda(cudaGetLastError());
		
		// No parallelization here, this simply prevents us from 
		// copying memory from/to host 20 times
		set_bnd_cuda<<<1, BLOCKSIZE>>>(N, b, gpu.a);
		checkCuda(cudaGetLastError());
	}
	to_host(N, x, x0, gpu);
}

void diffuse_cuda(int N, int b, fluid *x, fluid *x0, float diff, float dt, GPUSTATE gpu)
{
	float a = dt * diff * N * N;
	lin_solve_cuda(N, b, x, x0, a, 1 + 4 * a, gpu);
}

void dens_step_cuda(int N, fluid *x, fluid *x0, fluid *u, fluid *v, float diff, float dt, GPUSTATE gpu)
{
	add_source(N, x, x0, dt);
	SWAP(x0, x);
	diffuse_cuda(N, 0, x, x0, diff, dt, gpu);
	SWAP(x0, x);
	advect(N, 0, x, x0, u, v, dt);
}

void project_cuda(int N, fluid *u, fluid *v, fluid *p, fluid *div, GPUSTATE gpu)
{
	int i, j;

	FOR_EACH_CELL
	div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]) / N;
	p[IX(i, j)] = 0;
	END_FOR
	set_bnd(N, 0, div);
	set_bnd(N, 0, p);

	lin_solve_cuda(N, 0, p, div, 1, 4, gpu);

	FOR_EACH_CELL
	u[IX(i, j)] -= 0.5f * N * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
	v[IX(i, j)] -= 0.5f * N * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
	END_FOR
	set_bnd(N, 1, u);
	set_bnd(N, 2, v);
}

void vel_step_cuda(int N, fluid *u, fluid *v, fluid *u0, fluid *v0, float visc, float dt, GPUSTATE gpu)
{
	add_source(N, u, u0, dt);
	add_source(N, v, v0, dt);
	SWAP(u0, u);
	diffuse_cuda(N, 1, u, u0, visc, dt, gpu);
	SWAP(v0, v);
	diffuse_cuda(N, 2, v, v0, visc, dt, gpu);
	project_cuda(N, u, v, u0, v0, gpu);
	SWAP(u0, u);
	SWAP(v0, v);
	advect(N, 1, u, u0, u0, v0, dt);
	advect(N, 2, v, v0, u0, v0, dt);
	project_cuda(N, u, v, u0, v0, gpu);
}
