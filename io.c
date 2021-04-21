/*
  ======================================================================
   io.c --- Functions for saving/loading state of simulation to/from disk
  ----------------------------------------------------------------------
   Author : Jan Schutte (jan.schutte@student.uva.nl)
   Creation Date : Apr 21 2021

  =======================================================================
*/

#include <stdio.h>
#include "solver.h"

void save_to_disk(char* filename, int N, fluid *u, fluid *v, fluid *u_prev, fluid *v_prev, fluid *dens, fluid *dens_prev)
{
  // TODO: save more than the size
  FILE *fp;

  fp = fopen(filename, "w+");
  fprintf(fp, "N: %d\n", N);
  fclose(fp);
}