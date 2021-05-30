# Performance Engineering: Fluid Dynamics Project

Code for this implementation is based on work by Jos Stam:
"Real-Time Fluid Dynamics for Games". Proceedings of the Game Developer Conference, March 2003

Original source code can be found here:  
http://www.dgp.toronto.edu/people/stam/reality/Research/zip/CDROM_GDC03.zip

This CUDA version has been created by Jan Schutte.

# Requirements
- `gcc`
- `nvcc` (for cuda versions)
- `freeglut3-dev`
- `libgl-dev`
- `python3` (for generating plots)

# Compiling

Single precision:
```bash
make
```

Half precision:
```bash
make HALF_PRECISION=True
```

# Applications
Five applications are build when running make:
- `demo`: This is the (mostly) original sequential implementation
- `benchmark`: This benchmarks the sequential implementation
- `cuda-demo`: Final CUDA version of the application
- `cuda-benchmark`: This benchmarks the CUDA implementation
- `accuracy`: This application tests the accuracy of the half precision version (NOTE: do not forget to compile with half precision flag!)


# Running

## Interactive Demo
The demos can be run simply by starting the application, when no parameters are given default parameters are used. Information on interacting with the application is printed in the terminal.
The defaults for the `demo` application will still run fairly speedy on some systems, use these arguments to stress it a bit:

```bash
./demo 512 0.1 0 0 5 100
```

Similarly the CUDA version should handle it just fine:
```bash
./cuda-demo 512 0.1 0 0 5 100
```

## Accuracy
The `accuracy` application requires data from the `inputs/` and `outputs/` folders, the former contains start states of the simulation and the latter the state of the simulation after some time steps. This output has been generated using the single precision application.

```bash
./accuracy 128 0.1 0 0 5 100 50 inputs/128.fluid outputs/128_step_50.fluid
```

## Plots
The `plots.py` file will generate all plots from the report and print out all benchmarked parameters of the analytical models.
```bash
python3 plots.py
```