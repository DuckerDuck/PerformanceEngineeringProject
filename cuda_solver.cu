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
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
	fluid tmp = 0;
	
	// i and j start at 1
	i += 1;
	j += 1;

	if (i <= N && j <= N) {
		tmp = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)]));
		x[IX(i, j)] = tmp / c;
	}
}

__global__ void lin_solve_kernel_half(int N, fluid *x, fluid *x0, float a, float c) 
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	half ha = __float2half(a);
	half ca = __float2half(c);
	half tmp, k, l, m, o, p;
	
	// i  and j start at 1
	i += 1;
	j += 1;

	if (i <= N && j <= N) {
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


__global__ void set_bnd_kernel_a(int N, int b, fluid *x)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	i += 1;

	if (i <= N)
	{
		x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
		x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
		x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
		x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
	}
}

__global__ void set_bnd_kernel_b(int N, fluid *x)
{
	x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
	x[IX(0, N + 1)] = 0.5f * (x[IX(1, N + 1)] + x[IX(0, N)]);
	x[IX(N + 1, 0)] = 0.5f * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
	x[IX(N + 1, N + 1)] = 0.5f * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

void set_bnd_cuda(int N, int b, fluid *x)
{
	set_bnd_kernel_a<<<N/BLOCKSIZE + 1, BLOCKSIZE>>>(N, b, x);
	checkCuda(cudaGetLastError());
	
	// No parallelism here
	set_bnd_kernel_b<<<1, 1>>>(N, x);
	checkCuda(cudaGetLastError());
}

void lin_solve_cuda(int N, int b, fluid *x, fluid *x0, float a, float c)
{	
	int k;

	for (k = 0; k < 20; k++)
	{
		dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
    	dim3 dimGrid(N/BLOCKSIZE + 1, N/BLOCKSIZE + 1);
		LINSOLVE_KERNEL<<<dimGrid, dimBlock>>>(N, x, x0, a, c);
		checkCuda(cudaGetLastError());
		
		// No parallelization here, this simply prevents us from 
		// copying memory from/to host 20 times
		set_bnd_cuda(N, b, x);
		checkCuda(cudaGetLastError());
	}
	
}

void diffuse_cuda(int N, int b, fluid *x, fluid *x0, float diff, float dt)
{
	float a = dt * diff * N * N;
	lin_solve_cuda(N, b, x, x0, a, 1 + 4 * a);
}

__global__ void project_cuda_kernel_a(int N, fluid *u, fluid *v, fluid *u0, fluid *v0) {
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
	
	// i and j start at 1
	i += 1;
	j += 1;
	
	if (i <= N && j <= N)
	{
		v0[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]) / N;
		u0[IX(i, j)] = 0;
	}
}

__global__ void project_cuda_kernel_b(int N, fluid *u, fluid *v, fluid *p) {
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
	
	// i and j start at 1
	i += 1;
	j += 1;
	
	if (i <= N && j <= N)
	{
		u[IX(i, j)] -= 0.5f * N * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
		v[IX(i, j)] -= 0.5f * N * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
	}
}

void project_cuda(int N, fluid *u, fluid *v, fluid *p, fluid *div)
{
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(N/BLOCKSIZE + 1, N/BLOCKSIZE + 1);
	project_cuda_kernel_a<<<dimGrid, dimBlock>>>(N, u, v, p, div);
	checkCuda(cudaGetLastError());

	set_bnd_cuda(N, 0, div);
	set_bnd_cuda(N, 0, p);
	
	lin_solve_cuda(N, 0, p, div, 1, 4);

	project_cuda_kernel_b<<<dimGrid, dimBlock>>>(N, u, v, p);
	checkCuda(cudaGetLastError());

	set_bnd_cuda(N, 1, u);
	set_bnd_cuda(N, 2, v);
}

__global__ void advect_kernel(int N, fluid *d, fluid *d0, fluid *u, fluid *v, float dt)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int j = threadIdx.y + blockIdx.y * blockDim.y; 
	
	int i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1, dt0;
	dt0 = dt * N;
	// i and j start at 1
	i += 1;
	j += 1;
	
	if (i <= N && j <= N) 
	{
		x = i - dt0 * u[IX(i, j)];
		y = j - dt0 * v[IX(i, j)];
		if (x < 0.5f)
			x = 0.5f;
		if (x > N + 0.5f)
			x = N + 0.5f;
		i0 = (int)x;
		i1 = i0 + 1;
		if (y < 0.5f)
			y = 0.5f;
		if (y > N + 0.5f)
			y = N + 0.5f;
		j0 = (int)y;
		j1 = j0 + 1;
		s1 = x - i0;
		s0 = 1 - s1;
		t1 = y - j0;
		t0 = 1 - t1;
		d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
						s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
	}
	
}

void advect_cuda(int N, int b, fluid *d, fluid *d0, fluid *u, fluid *v, float dt)
{
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(N/BLOCKSIZE + 1, N/BLOCKSIZE + 1);
	advect_kernel<<<dimGrid, dimBlock>>>(N, d, d0, u, v, dt);
	checkCuda(cudaGetLastError());

	set_bnd_cuda(N, b, d);
	checkCuda(cudaGetLastError());
}

__global__ void add_source_kernel(int size, fluid *x, fluid *s, float dt) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		x[i] += dt * s[i];
	}
}

void add_source_cuda(int N, fluid *x, fluid *s, float dt)
{
	int size = (N + 2) * (N + 2);
	add_source_kernel<<<size/BLOCKSIZE + 1, BLOCKSIZE>>>(size, x, s, dt);
	checkCuda(cudaGetLastError());
}

void step_cuda(int N, fluid *u, fluid *v, fluid *u0, fluid *v0, fluid *x, fluid *x0,  float visc, float dt, float diff, GPUSTATE gpu)
{
	int size = (N + 2) * (N + 2) * sizeof(fluid);
	checkCuda(cudaMemcpyAsync(gpu.dens, x, size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyAsync(gpu.dens_prev, x0, size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyAsync(gpu.u, u, size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyAsync(gpu.u_prev, u0, size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyAsync(gpu.v, v, size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyAsync(gpu.v_prev, v0, size, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	vel_step_cuda(N, gpu.u, gpu.v, gpu.u_prev, gpu.v_prev, visc, dt);
	dens_step_cuda(N, gpu.dens, gpu.dens_prev, gpu.u, gpu.v, diff, dt);

	checkCuda(cudaMemcpyAsync(x, gpu.dens, size, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpyAsync(x0, gpu.dens_prev, size, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpyAsync(u, gpu.u, size, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpyAsync(u0, gpu.u_prev, size, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpyAsync(v, gpu.v, size, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpyAsync(v0, gpu.v_prev, size, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

}


void dens_step_cuda(int N, fluid *x, fluid *x0, fluid *u, fluid *v, float diff, float dt)
{
	add_source_cuda(N, x, x0, dt);
	SWAP(x0, x);
	diffuse_cuda(N, 0, x, x0, diff, dt);
	SWAP(x0, x);
	advect_cuda(N, 0, x, x0, u, v, dt);
}


void vel_step_cuda(int N, fluid *u, fluid *v, fluid *u0, fluid *v0, float visc, float dt)
{
	add_source_cuda(N, u, u0, dt);
	add_source_cuda(N, v, v0, dt);
	SWAP(u0, u);
	diffuse_cuda(N, 1, u, u0, visc, dt);
	SWAP(v0, v);
	diffuse_cuda(N, 2, v, v0, visc, dt);
	project_cuda(N, u, v, u0, v0);
	SWAP(u0, u);
	SWAP(v0, v);
	advect_cuda(N, 1, u, u0, u0, v0, dt);
	advect_cuda(N, 2, v, v0, u0, v0, dt);
	project_cuda(N, u, v, u0, v0);
}
