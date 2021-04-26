#ifndef _SOLVER_H
#define _SOLVER_H

#define IX(i, j) ((i) + (N.w + 2) * (j))

#ifndef DATATYPE
typedef float fluid;
#else
typedef DATATYPE fluid;
#endif

// Structure for simulation size
typedef struct {
	int h, w;
} grid_size;

#endif