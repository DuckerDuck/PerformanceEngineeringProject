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

#define WRITE(x)                         \
  for (int i = 0; i < N + 2; i++)        \
  {                                      \
    for (int j = 0; j < N + 2; j++)      \
    {                                    \
      fprintf(fp, "%.4f ", x[IX(i, j)]); \
    }                                    \
    fprintf(fp, "\n");                   \
  }

#define READ(x)                                             \
  for (int i = 0; i < N + 2; i++)                           \
  {                                                         \
    for (int j = 0; j < N + 2; j++)                         \
    {                                                       \
      read = fscanf(fp, "%f ", &val);                       \
      if (read != 1)                                        \
      {                                                     \
        printf("Could not read values from fluid file!\n"); \
        return;                                             \
      }                                                     \
      x[IX(i, j)] = (fluid)val;                             \
    }                                                       \
    read = fscanf(fp, "\n");                                       \
  }

void save_to_disk(char *filename, int N, fluid *u, fluid *v, fluid *u_prev, fluid *v_prev, fluid *dens, fluid *dens_prev)
{
  FILE *fp;

  fp = fopen(filename, "w+");
  fprintf(fp, "N: %d\n", N);

  WRITE(u);
  WRITE(u_prev);
  WRITE(v);
  WRITE(v_prev);
  WRITE(dens);
  WRITE(dens_prev);

  fclose(fp);
}

void read_from_disk(char *filename, int N, fluid *u, fluid *v, fluid *u_prev, fluid *v_prev, fluid *dens, fluid *dens_prev)
{
  FILE *fp;
  int file_N = -1;
  int read = 0;
  float val;

  fp = fopen(filename, "r");
  read = fscanf(fp, "N: %d\n", &file_N);
  if (read != 1)
  {
    printf("Could not read value of N from fluid file!\n");
    return;
  }

  READ(u);
  READ(u_prev);
  READ(v);
  READ(v_prev);
  READ(dens);
  READ(dens_prev);

  fclose(fp);
}