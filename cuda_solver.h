#ifndef _CUDA_SOLVER_H
#define _CUDA_SOLVER_H
#include "solver.h"

#ifdef HALF_PRECISION
#define LINSOLVE_KERNEL lin_solve_kernel_half
#else
#define LINSOLVE_KERNEL lin_solve_kernel
#endif

#define BLOCKSIZE 16

typedef struct GPUSTATE {
	fluid *u, *v, *u_prev, *v_prev;
	fluid *dens, *dens_prev;
} GPUSTATE;

void step_cuda(int N, fluid *u, fluid *v, fluid *u0, fluid *v0, fluid *x, fluid *x0,  float visc, float dt, float diff, GPUSTATE gpu);

void vel_step_cuda(int N, fluid *u, fluid *v, fluid *u0, fluid *v0, float visc, float dt, GPUSTATE gpu);
void dens_step_cuda(int N, fluid *x, fluid *x0, fluid *u, fluid *v, float diff, float dt, GPUSTATE gpu);

void lin_solve_cuda(int N, int b, fluid *x, fluid *x0, float a, float c);
void diffuse_cuda(int N, int b, fluid *x, fluid *x0, float diff, float dt, GPUSTATE gpu);
void project_cuda(int N, fluid *u, fluid *v, fluid *p, fluid *div, GPUSTATE gpu);
void advect_cuda(int N, int b, fluid *d, fluid *d0, fluid *u, fluid *v, float dt, GPUSTATE gpu);



void checkCuda(cudaError_t result);
#endif