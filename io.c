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
  for (int i = 0; i < N.w + 2; i++)        \
  {                                      \
    for (int j = 0; j < N.h + 2; j++)      \
    {                                    \
      fprintf(fp, "%.10f ", x[IX(i, j)]); \
    }                                    \
    fprintf(fp, "\n");                   \
  }

#define READ(x)                                             \
  for (int i = 0; i < file_W + 2; i++)                           \
  {                                                         \
    for (int j = 0; j < file_H + 2; j++)                         \
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

void save_to_disk(char *filename, grid_size N, fluid *u, fluid *v, fluid *u_prev, fluid *v_prev, fluid *dens, fluid *dens_prev)
{
  FILE *fp;

  fp = fopen(filename, "w+");
  fprintf(fp, "N: %d, %d\n", N.w, N.h);

  WRITE(u);
  WRITE(u_prev);
  WRITE(v);
  WRITE(v_prev);
  WRITE(dens);
  WRITE(dens_prev);

  fclose(fp);
}

void read_from_disk(char *filename, grid_size N, fluid *u, fluid *v, fluid *u_prev, fluid *v_prev, fluid *dens, fluid *dens_prev)
{
  FILE *fp;
  int file_W = -1;
  int file_H = -1;
  int read = 0;
  float val;

  fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Could not read file \"%s\"!", filename);
    return;
  }
  read = fscanf(fp, "N: %d, %d\n", &file_W, &file_H);
  if (read != 2)
  {
    printf("Could not read value of N from fluid file!\n");
    return;
  }

  if (file_W > N.w || file_H > N.h) {
    printf("W and H parameters are bigger than current program config!\n");
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