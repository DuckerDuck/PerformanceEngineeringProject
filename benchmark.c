/*
  ======================================================================
   benchmark.c --- Benchmark fluid solver
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

#include "solver.h"
#include "io.h"

/* external definitions (from solver.c) */
extern void dens_step(int N, fluid *x, fluid *x0, fluid *u, fluid *v, float diff, float dt);
extern void vel_step(int N, fluid *u, fluid *v, fluid *u0, fluid *v0, float visc, float dt);

/* Simulation State */
static int N;
static float dt, diff, visc;
static float force, source;
static fluid *u, *v, *u_prev, *v_prev;
static fluid *dens, *dens_prev;

/* Benchmark State */
static int steps;
static int runs;

/* Timing Functions */

// Return number of seconds since unix Epoch
double get_time()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec*1e-6;
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

static void step(void)
{
	vel_step(N, u, v, u_prev, v_prev, visc, dt);
	dens_step(N, dens, dens_prev, u, v, diff, dt);
}

static void benchmark(char* start_state, int file_N)
{
	double start_time, end_time, total_time;
	total_time = 0;
	N = file_N;
	if (u) {
		free_data();
	}
	if (!allocate_data()) {
		fprintf(stderr, "Could not allocate data for run\n");
	}
	clear_data();

	printf("Start state: %s\n", start_state);
	
	for (int r = 0; r < runs; r++)
	{
		read_from_disk(start_state, file_N, u, v, u_prev, v_prev, dens, dens_prev);

		start_time = get_time();
		for (int s = 0; s < steps; s++)
		{
			step();
		}
		end_time = get_time();
		total_time += (end_time - start_time);
	}

	double step_time_s = (total_time/(runs*steps));
	printf("\tTotal time: %lf s, time per step: %lf ms, frames per second: %lf \n", total_time, step_time_s * 1e3, 1.0 / step_time_s);
}

int main(int argc, char **argv)
{

	if (argc != 1 && argc != 9)
	{
		fprintf(stderr, "usage : %s N dt diff visc force source\n", argv[0]);
		fprintf(stderr, "where:\n");
		fprintf(stderr, "\t N      : grid resolution\n");
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
		N = 64;
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
		N = atoi(argv[1]);
		dt = atof(argv[2]);
		diff = atof(argv[3]);
		visc = atof(argv[4]);
		force = atof(argv[5]);
		source = atof(argv[6]);
		steps = atoi(argv[7]);
		runs = atoi(argv[8]);
	}

	printf("Arguments : N=%d dt=%g diff=%g visc=%g force=%g source=%g steps=%d runs=%d\n",
			N, dt, diff, visc, force, source, steps, runs);

	benchmark("inputs/32.fluid", 32);
	benchmark("inputs/64.fluid", 64);
	benchmark("inputs/96.fluid", 96);
	benchmark("inputs/128.fluid", 128);
	benchmark("inputs/256.fluid", 256);
	benchmark("inputs/512.fluid", 512);
	
	free_data();

	exit(0);
}
