/*
  ======================================================================
   cuda_benchmark.c --- Benchmark fluid solver
  ----------------------------------------------------------------------
   Author : Jan Schutte (jan.schutte@student.uva.nl)
   Creation Date : Apr 26 2021

   Description:

	Interface for creating reproducible benchmark results

  =======================================================================
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <string.h>
#include <cmath>

#include "solver.h"
#include "io.h"

#include "cuda_solver.h"


/* Simulation State */
static int N;
static float dt, diff, visc;
static float force, source;
static fluid *u, *v, *u_prev, *v_prev;
static fluid *dens, *dens_prev;

/* Device Simulation State */
static GPUSTATE gpu;

/* Correct end state of simulation */
static fluid *u_correct, *v_correct, *u_prev_correct, *v_prev_correct;
static fluid *dens_correct, *dens_prev_correct;

/* State */
static int steps;
static char *input;
static char *output;

/* This flag makes causes the end state to be written 
*  to the output file */
#define COMPARE_ONLY	

/* When generating data, use original step function */
#ifdef COMPARE_ONLY
#define STEPFUN step
#else
#define STEPFUN original_step
#endif

static void free_data(void)
{
	if (u)
		cudaFreeHost(u);
	if (v)
		cudaFreeHost(v);
	if (u_prev)
		cudaFreeHost(u_prev);
	if (v_prev)
		cudaFreeHost(v_prev);
	if (dens)
		cudaFreeHost(dens);
	if (dens_prev)
		cudaFreeHost(dens_prev);
}

static int allocate_data(void)
{
	int size = (N + 2) * (N + 2) * sizeof(fluid);

	checkCuda(cudaMallocHost((void**)&u, size));
	checkCuda(cudaMallocHost((void**)&v, size));
	checkCuda(cudaMallocHost((void**)&u_prev, size));
	checkCuda(cudaMallocHost((void**)&v_prev, size));
	checkCuda(cudaMallocHost((void**)&dens, size));
	checkCuda(cudaMallocHost((void**)&dens_prev, size));

	checkCuda(cudaMallocHost((void**)&u_correct, size));
	checkCuda(cudaMallocHost((void**)&v_correct, size));
	checkCuda(cudaMallocHost((void**)&u_prev_correct, size));
	checkCuda(cudaMallocHost((void**)&v_prev_correct, size));
	checkCuda(cudaMallocHost((void**)&dens_correct, size));
	checkCuda(cudaMallocHost((void**)&dens_prev_correct, size));

	if (!u || !v || !u_prev || !v_prev || !dens || !dens_prev)
	{
		fprintf(stderr, "cannot allocate data\n");
		return (0);
	}

	return (1);
}

static int cuda_allocate_data(void)
{
	int size = (N + 2) * (N + 2) * sizeof(fluid);
	
	gpu.u = NULL;
	checkCuda(cudaMalloc((void **) &gpu.u, size));
	
	gpu.v = NULL;
	checkCuda(cudaMalloc((void **) &gpu.v, size));
	
	gpu.u_prev = NULL;
	checkCuda(cudaMalloc((void **) &gpu.u_prev, size));

	gpu.v_prev = NULL;
	checkCuda(cudaMalloc((void **) &gpu.v_prev, size));

	gpu.dens = NULL;
	checkCuda(cudaMalloc((void **) &gpu.dens, size));

	gpu.dens_prev = NULL;
	checkCuda(cudaMalloc((void **) &gpu.dens_prev, size));

	if (!gpu.u || !gpu.v || !gpu.u_prev || 
		!gpu.v_prev || !gpu.dens || !gpu.dens_prev ) {
		return 0;
	}
	return 1;
}

static void step(void)
{
	step_cuda(N, u, v, u_prev, v_prev, dens, dens_prev, visc, dt, diff, gpu);
}

static void original_step(void)
{
	vel_step(N, u, v, u_prev, v_prev, visc, dt);
	dens_step(N, dens, dens_prev, u, v, diff, dt);
}


static void mse()
{
	double u_err, v_err, dens_err;
	int size = (N + 2) * (N + 2);
	u_err = v_err = dens_err = 0.0;
	
	for (int i = 0; i < N + 2; i++)
	{
		for (int j = 0; j < N + 2; j++)
		{
			u_err += pow(u_correct[IX(i, j)] - u[IX(i, j)], 2);
			v_err += pow(v_correct[IX(i, j)] - v[IX(i, j)], 2);
			dens_err += pow(dens_correct[IX(i, j)] - dens[IX(i, j)], 2);
		}
	}

	printf("Error: \n\tu: %lf\n\tv: %lf\n\tdens: %lf", u_err/size, v_err/size, dens_err/size);
}

static void load_and_simulate()
{
	if (!allocate_data())
	{
		fprintf(stderr, "Could not allocate data for run\n");
		return;
	}

	if (!cuda_allocate_data()) {
		fprintf(stderr, "Could not allocate data on GPU device for run\n");
		return;
	}

	printf("reading start state from: %s\n", input);
	read_from_disk(input, N, u, v, u_prev, v_prev, dens, dens_prev);

	#ifdef COMPARE_ONLY
	printf("reading correct values from: %s\n", output);
	read_from_disk(output, N, u_correct, v_correct, u_prev_correct, v_prev_correct, dens_correct, dens_prev_correct);
	#endif

	for (int s = 0; s < steps; s++)
	{
		STEPFUN();
	}

	// For saving the correct output, don't forget to compile with the biggest datatype!
	#ifndef COMPARE_ONLY
	printf("Saving end state to: %s\n", output);
	save_to_disk(output, N, u, v, u_prev, v_prev, dens, dens_prev);
	#else
	mse();
	#endif

}


int main(int argc, char **argv)
{

	if (argc != 10)
	{
		fprintf(stderr, "usage : %s N dt diff visc force source steps input output\n", argv[0]);
		fprintf(stderr, "where:\n");
		fprintf(stderr, "\t dt     : time step\n");
		fprintf(stderr, "\t diff   : diffusion rate of the density\n");
		fprintf(stderr, "\t visc   : viscosity of the fluid\n");
		fprintf(stderr, "\t force  : scales the mouse movement that generate a force\n");
		fprintf(stderr, "\t source : amount of density that will be deposited\n");
		fprintf(stderr, "\t steps  : Number of steps to run simulation for\n");
		fprintf(stderr, "\t input  : Fluid file starting state\n");
		fprintf(stderr, "\t output : Fluid file with correct end state\n");
		exit(1);
	}

	if (argc != 1)
	{
		N = atoi(argv[1]);
		dt = atof(argv[2]);
		diff = atof(argv[3]);
		visc = atof(argv[4]);
		force = atof(argv[5]);
		source = atof(argv[6]);
		steps = atoi(argv[7]);
		input = argv[8];
		output = argv[9];
	}

	printf("Arguments: N=%d dt=%g diff=%g visc=%g force=%g source=%g steps=%d input=%s output=%s\n",
		   N, dt, diff, visc, force, source, steps, input, output);

	load_and_simulate();

	free_data();

	exit(0);
}
