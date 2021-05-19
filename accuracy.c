/*
  ======================================================================
   accuracy.c --- Check accuracy of fluid solver
  ----------------------------------------------------------------------
   Author : Jan Schutte (jan.schutte@student.uva.nl)
   Creation Date : May 19 2021

   Description:

	Interface for check the accuracy of fluid simulator

  =======================================================================
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <string.h>

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

/* Correct end state of simulation */
static fluid *u_correct, *v_correct, *u_prev_correct, *v_prev_correct;
static fluid *dens_correct, *dens_prev_correct;

/* State */
static int steps;
static char *input;
static char *output;

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

static int allocate_data(void)
{
	int size = (N + 2) * (N + 2);

	u = (fluid *)malloc(size * sizeof(fluid));
	v = (fluid *)malloc(size * sizeof(fluid));
	u_prev = (fluid *)malloc(size * sizeof(fluid));
	v_prev = (fluid *)malloc(size * sizeof(fluid));
	dens = (fluid *)malloc(size * sizeof(fluid));
	dens_prev = (fluid *)malloc(size * sizeof(fluid));

	u_correct = (fluid *)malloc(size * sizeof(fluid));
	v_correct = (fluid *)malloc(size * sizeof(fluid));
	u_prev_correct = (fluid *)malloc(size * sizeof(fluid));
	v_prev_correct = (fluid *)malloc(size * sizeof(fluid));
	dens_correct = (fluid *)malloc(size * sizeof(fluid));
	dens_prev_correct = (fluid *)malloc(size * sizeof(fluid));

	if (!u || !v || !u_prev || !v_prev || !dens || !dens_prev || !u_correct || !v_correct || !u_prev_correct || !v_prev_correct || !dens_correct || !dens_prev_correct)
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

static double sse()
{
	double err = 0;
	for (int i = 0; i < N + 2; i++)
	{
		for (int j = 0; j < N + 2; j++)
		{
			err += u[IX(i, j)] - u_correct[IX(i, j)];
			err += v[IX(i, j)] - v_correct[IX(i, j)];
		}
	}

	return err;
}

static void load_and_simulate()
{
	if (u)
		free_data();

	if (!allocate_data())
	{
		fprintf(stderr, "Could not allocate data for run\n");
		return;
	}

	read_from_disk(input, N, u, v, u_prev, v_prev, dens, dens_prev);
	read_from_disk(output, N, u_correct, v_correct, u_prev_correct, v_prev_correct, dens_correct, dens_prev_correct);

	for (int s = 0; s < steps; s++)
	{
		step();
	}

	// For generating the correct output, don't forget to compile with the biggest datatype!
	// save_to_disk(output, N, u, v, u_prev, v_prev, dens, dens_prev);

	printf("Err: %lf", sse());
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
