#ifndef _SOLVER_H
#define _SOLVER_H

#define IX(i, j) ((i) + (N + 2) * (j))

#define SWAP(x0, x)      \
	{                    \
		fluid *tmp = x0; \
		x0 = x;          \
		x = tmp;         \
	}
#define FOR_EACH_CELL            \
	for (i = 1; i <= N; i++)     \
	{                            \
		for (j = 1; j <= N; j++) \
		{
#define END_FOR \
	}           \
	}

#ifndef DATATYPE
typedef float fluid;
#else
typedef DATATYPE fluid;
#endif

void dens_step(int N, fluid *x, fluid *x0, fluid *u, fluid *v, float diff, float dt);
void vel_step(int N, fluid *u, fluid *v, fluid *u0, fluid *v0, float visc, float dt);

void project(int N, fluid *u, fluid *v, fluid *p, fluid *div);
void advect(int N, int b, fluid *d, fluid *d0, fluid *u, fluid *v, float dt);
void add_source(int N, fluid *x, fluid *s, float dt);
void lin_solve(int N, int b, fluid *x, fluid *x0, float a, float c);

#endif