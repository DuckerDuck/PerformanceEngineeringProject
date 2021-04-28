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

#endif