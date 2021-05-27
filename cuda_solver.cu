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


void lin_solve_cuda(int N, int b, fluid *x, fluid *x0, float a, float c)
{	
	int k;

	for (k = 0; k < 20; k++)
	{
		LINSOLVE_KERNEL<<<N/BLOCKSIZE + 1, BLOCKSIZE>>>(N, x, x0, a, c);
		checkCuda(cudaGetLastError());
		
		// No parallelization here, this simply prevents us from 
		// copying memory from/to host 20 times
		set_bnd_cuda<<<1, BLOCKSIZE>>>(N, b, x);
		checkCuda(cudaGetLastError());
	}
	
}

void diffuse_cuda(int N, int b, fluid *x, fluid *x0, float diff, float dt, GPUSTATE gpu)
{
	float a = dt * diff * N * N;	
	int size = (N + 2) * (N + 2) * sizeof(fluid);

	checkCuda(cudaMemcpy(gpu.dens, x, size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(gpu.dens_prev, x0, size, cudaMemcpyHostToDevice));

	lin_solve_cuda(N, b, gpu.dens, gpu.dens_prev, a, 1 + 4 * a);
	
	checkCuda(cudaMemcpy(x, gpu.dens, size, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(x0, gpu.dens_prev, size, cudaMemcpyDeviceToHost));

}

void dens_step_cuda(int N, fluid *x, fluid *x0, fluid *u, fluid *v, float diff, float dt, GPUSTATE gpu)
{
	add_source(N, x, x0, dt);
	SWAP(x0, x);
	diffuse_cuda(N, 0, x, x0, diff, dt, gpu);
	SWAP(x0, x);
	advect(N, 0, x, x0, u, v, dt);
}

__global__ void project_cuda_kernel_a(int N, fluid *u, fluid *v, fluid *u0, fluid *v0) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j;
	// i starts at 1
	i += 1;
	
	if (i <= N) {
		for (j = 1; j <= N; j++) 
		{
			v0[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]) / N;
			u0[IX(i, j)] = 0;
		}
	}
}

__global__ void project_cuda_kernel_b(int N, fluid *u, fluid *v, fluid *u0) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j;
	// i starts at 1
	i += 1;
	
	if (i <= N) {
		for (j = 1; j <= N; j++) 
		{
			u[IX(i, j)] -= 0.5f * N * (u0[IX(i + 1, j)] - u0[IX(i - 1, j)]);
			v[IX(i, j)] -= 0.5f * N * (u0[IX(i, j + 1)] - u0[IX(i, j - 1)]);
		}
	}
}

void project_cuda(int N, fluid *u, fluid *v, fluid *u0, fluid *v0, GPUSTATE gpu)
{
	int size = (N + 2) * (N + 2) * sizeof(fluid);
	checkCuda(cudaMemcpy(gpu.u, u, size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(gpu.v, v, size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(gpu.u_prev, u0, size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(gpu.v_prev, v0, size, cudaMemcpyHostToDevice));

	project_cuda_kernel_a<<<N/BLOCKSIZE, BLOCKSIZE>>>(N, gpu.u, gpu.v, gpu.u_prev, gpu.v_prev);
	checkCuda(cudaGetLastError());

	set_bnd_cuda<<<1, BLOCKSIZE>>>(N, 0, gpu.v_prev);
	set_bnd_cuda<<<1, BLOCKSIZE>>>(N, 0, gpu.u_prev);
	
	lin_solve_cuda(N, 0, gpu.u_prev, gpu.v_prev, 1, 4);

	project_cuda_kernel_b<<<N/BLOCKSIZE, BLOCKSIZE>>>(N, gpu.u, gpu.v, gpu.u_prev);
	checkCuda(cudaGetLastError());

	set_bnd_cuda<<<1, BLOCKSIZE>>>(N, 1, gpu.u);
	set_bnd_cuda<<<1, BLOCKSIZE>>>(N, 2, gpu.v);

	checkCuda(cudaMemcpy(u0, gpu.u_prev, size, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(v0, gpu.v_prev, size, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(u, gpu.u, size, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(v, gpu.v, size, cudaMemcpyDeviceToHost));
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
