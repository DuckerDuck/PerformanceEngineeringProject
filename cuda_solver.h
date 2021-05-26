#include "solver.h"

#ifndef _CUDA_SOLVER_H
#define _CUDA_SOLVER_H

void lin_solve_cuda(int N, int b, fluid *x, fluid *x0, float a, float c);

#endif