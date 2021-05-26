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

/* Benchmark State */
static int steps;
static int runs;

/* Timing Functions */

// Return number of seconds since unix Epoch
double get_time()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + t.tv_usec * 1e-6;
}

static void free_data(void)
{
	if (u)
		free(u);
	if (v)
		free(v);
	if (u_prev)
		free(u_prev);
	if (v_prev)
		free(v_prev);
	if (dens)
		free(dens);
	if (dens_prev)
		free(dens_prev);
}

static void clear_data(void)
{
	int i, size = (N + 2) * (N + 2);

	for (i = 0; i < size; i++)
	{
		u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
	}
}

static int allocate_data(void)
{
	int size = (N + 2) * (N + 2);

	u = (fluid *)malloc(size * sizeof(fluid));
	v = (fluid *)malloc(size * sizeof(fluid));
	u_prev = (fluid *)malloc(size * sizeof(fluid));
	v_prev = (fluid *)malloc(size * sizeof(fluid));
	dens = (fluid *)malloc(size * sizeof(fluid));
	dens_prev = (fluid *)malloc(size * sizeof(fluid));

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
	
	gpu.a = NULL;
	checkCuda(cudaMalloc((void **) &gpu.a, size));
	
	gpu.b = NULL;
	checkCuda(cudaMalloc((void **) &gpu.b, size));
	
	// gpu.u_prev = NULL;
	// checkCuda(cudaMalloc((void **) &gpu.u_prev, size));

	// gpu.v_prev = NULL;
	// checkCuda(cudaMalloc((void **) &gpu.v_prev, size));

	// gpu.dens = NULL;
	// checkCuda(cudaMalloc((void **) &gpu.dens, size));

	// gpu.dens_prev = NULL;
	// checkCuda(cudaMalloc((void **) &gpu.dens_prev, size));

	// if (!gpu.u || !gpu.v || !gpu.u_prev || 
	// 	!gpu.v_prev || !gpu.dens || !gpu.dens_prev ) {
	// 	return 0;
	// }
	if (!gpu.a || !gpu.b) 
	{
		return 0;
	}
	return 1;
}

static void step(void)
{
	vel_step(N, u, v, u_prev, v_prev, visc, dt);
	dens_step(N, dens, dens_prev, u, v, diff, dt);
}

float random_float(float min, float max)
{
	float number;
	number = (float)rand() / (float)(RAND_MAX / (max - min));
	number += min;
	return number;
}

static void set_random_state()
{
	int i, j;
	const float min = -1;
	const float max = 1;

	FOR_EACH_CELL
		u[IX(i, j)] = random_float(min, max);
		v[IX(i, j)] = random_float(min, max);
		u_prev[IX(i, j)] = random_float(min, max);
		v_prev[IX(i, j)] = random_float(min, max);
		dens[IX(i, j)] = random_float(min * 10, max * 10);
		dens_prev[IX(i, j)] = random_float(min * 10, max * 10);
	END_FOR
}

static void benchmark(int file_N)
{
	double start_time, end_time, total_time, lin_solve_time, advect_time, project_time, add_source_time;
	int s = 0;
	total_time = lin_solve_time = advect_time = project_time = add_source_time = 0.0;
	
	N = file_N;

	if (u)
		free_data();
	
	if (!allocate_data()) {
		fprintf(stderr, "Could not allocate data for run\n");
		return;
	}

	if (!cuda_allocate_data()) {
		fprintf(stderr, "Could not allocate data on GPU device for run\n");
		return;
	}
	
	clear_data();

	printf("N: %d, ", N);

	for (int r = 0; r < runs; r++)
	{	
		set_random_state();
		// read_from_disk(start_state, file_N, u, v, u_prev, v_prev, dens, dens_prev);
	
		// Time for whole application
		start_time = get_time();
		for (s = 0; s < steps; s++)
		{
			step();
		}
		end_time = get_time();
		total_time += end_time - start_time;

		// Time for project function
		start_time = get_time();
		for (s = 0; s < steps; s++)
		{
			project(N, u, v, u_prev, v_prev);
		}
		end_time = get_time();
		project_time += end_time - start_time;

		// Time for lin solve function
		start_time = get_time();
		for (s = 0; s < steps; s++)
		{
			lin_solve_cuda(N, 0, dens, dens_prev, 1, 4, gpu);
		}
		end_time = get_time();
		lin_solve_time += end_time - start_time;

		// Time for advect function
		start_time = get_time();
		for (s = 0; s < steps; s++)
		{
			advect(N, 0, dens, dens_prev, u, v, dt);
		}
		end_time = get_time();
		advect_time += end_time - start_time;

		// Time for add_source function
		start_time = get_time();
		for (s = 0; s < steps; s++)
		{
			add_source(N, u, u_prev, dt);
		}
		end_time = get_time();
		add_source_time += end_time - start_time;
		
	}

	double step_time_total_s = (total_time / (runs * steps));
	double step_time_advect_s = (advect_time / (runs * steps));
	double step_time_lin_solve_s = (lin_solve_time / (runs * steps));
	double step_time_add_source_s = (add_source_time / (runs * steps));
	double step_time_project_s = (project_time / (runs * steps));
	printf("total: %lf s, total step: %lf ms, frames per second: %lf, ", total_time, step_time_total_s * 1e3, 1.0 / step_time_total_s);
	printf("advect: %lf ms, ", step_time_advect_s * 1e3);
	printf("lin_solve: %lf ms, ", step_time_lin_solve_s * 1e3);
	printf("add_source: %lf ms, ", step_time_add_source_s * 1e3);
	printf("project: %lf ms \n", step_time_project_s * 1e3);
}

int main(int argc, char **argv)
{
	if (argc != 1 && argc != 8)
	{
		fprintf(stderr, "usage : %s N dt diff visc force source\n", argv[0]);
		fprintf(stderr, "where:\n");
		fprintf(stderr, "\t dt     : time step\n");
		fprintf(stderr, "\t diff   : diffusion rate of the density\n");
		fprintf(stderr, "\t visc   : viscosity of the fluid\n");
		fprintf(stderr, "\t force  : scales the mouse movement that generate a force\n");
		fprintf(stderr, "\t source : amount of density that will be deposited\n");
		fprintf(stderr, "\t steps  : Number of steps to run simulation for\n");
		fprintf(stderr, "\t runs   : Number of times to a single simulation\n");
		exit(1);
	}

	if (argc == 1)
	{
		dt = 0.1f;
		diff = 0.0f;
		visc = 0.0f;
		force = 5.0f;
		source = 100.0f;
		steps = 10;
		runs = 10;
	}
	else
	{
		dt = atof(argv[1]);
		diff = atof(argv[2]);
		visc = atof(argv[3]);
		force = atof(argv[4]);
		source = atof(argv[5]);
		steps = atoi(argv[6]);
		runs = atoi(argv[7]);
	}

	printf("Arguments: dt=%g diff=%g visc=%g force=%g source=%g steps=%d runs=%d\n",
		   dt, diff, visc, force, source, steps, runs);

	for (int i = 512; i <= 1024; i += 64)
		benchmark(i);
	
	free_data();

	exit(0);
}
