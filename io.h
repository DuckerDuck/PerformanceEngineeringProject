#ifndef _IO_H
#define _IO_H

#include "solver.h"

void save_to_disk(char* filename, grid_size N, fluid *u, fluid *v, fluid *u_prev, fluid *v_prev, fluid *dens, fluid *dens_prev);
void read_from_disk(char *filename, grid_size N, fluid *u, fluid *v, fluid *u_prev, fluid *v_prev, fluid *dens, fluid *dens_prev);

#endif