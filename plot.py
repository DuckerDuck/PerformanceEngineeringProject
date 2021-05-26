import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_output(file: Path, x_key='N', y_key='time per step'):

    x = []
    y = []
    
    with open(file, 'r') as f:
        output = f.readlines()

    for line in output:
        metrics = line.split(',')
        for m in metrics:
            key, value_unit = m.split(':')
            key = key.strip()
            value_unit = value_unit.strip()

            if len(value_unit.split(' ')) == 1:
                value = value_unit
            else:
                value, unit = value_unit.split(' ')

            if '.' in value:
                value = float(value)
            else:
                value = int(value)

            if key == x_key:
                x.append(value)
            if key == y_key:
                y.append(value)

    return x, y

def lin_solve(n, ls_c, p, k=20):
    """ Analytical model for lin_solve function
    n: grid size
    ls_c: lin solve kernel constant
    k: Number of iterations of linear solve algorithm, should be the same value as in solver.c:lin_solve
    """
    return (k * n * n * ls_c) / p

def project(n, ls_c, proj_c, p):
    """ Analytical model for project function
    n: grid size
    ls_c: lin solve kernel constant
    proj_c: project kernel constant
    """
    return lin_solve(n, ls_c, p) + (2 * n * n * proj_c)

def add_source(n, src_c):
    """ Analytical model for add_source function
    n: grid size
    src_c: add_source kernel constant
    """
    return (n*n*src_c) 

def advect(n, adv_c):
    """ Analytical model for advect function
    n: grid size
    adv_c: advect kernel constant
    """
    return n*n*adv_c

def total(n, adv_c, src_c, proj_c, ls_c, p=1):
    """ Full analytical model """
    return 3 * add_source(n, adv_c) + \
           2 * project(n, ls_c, proj_c, p) + \
           3 * lin_solve(n, ls_c, p) + \
           3 * advect(n, adv_c)

def plot():
    n, y_total = parse_output(Path('./output'), 'N', 'total step')
    _, y_advect = parse_output(Path('./output'), 'N', 'advect')
    _, y_linsolve = parse_output(Path('./output'), 'N', 'lin_solve')
    _, y_project = parse_output(Path('./output'), 'N', 'project')
    _, y_source = parse_output(Path('./output'), 'N', 'add_source')
    
    # Fit parameters of analytical model
    ls_c = np.mean([y / (20*nn*nn) for y, nn in zip(y_linsolve, n)])
    proj_c = np.mean([(y - lin_solve(nn, ls_c, 1)) / (2 * nn * nn) for y, nn in zip(y_project, n)])
    src_c = np.mean([y / (nn*nn) for y, nn in zip(y_source, n)])
    adv_c = np.mean([y / (nn*nn) for y, nn in zip(y_advect, n)])
    
    
    plt.figure()
    plt.plot(n, y_total, label='total')
    plt.plot(n, [total(nn, adv_c, src_c, proj_c, ls_c) for nn in n], label='analytical p=1')
    
    # plt.plot(n, y_advect, label='advect')
    # plt.plot(n, [advect(nn, adv_c) for nn in n], label='advect fit')
    
    # plt.plot(n, y_linsolve, label='lin_solve')
    # plt.plot(n, [lin_solve(nn, ls_c) for nn in n], label='lin_solve fit')
    
    # plt.plot(n, y_project, label='project')
    # plt.plot(n, [project(nn, ls_c, proj_c) for nn in n], label='project fit')
    
    # plt.plot(n, y_source, label='add_source')
    # plt.plot(n, [add_source(nn, src_c) for nn in n], label='add_source fit')
    
    # Plot what we want
    plt.plot(n, [16.7] * len(n), label='60 FPS', linestyle='--')
    # plt.plot(n, [6.9] * len(n), label='144 FPS', linestyle='--')

    plt.ylabel('Step time (ms)')
    plt.xlabel('Grid size (N)')
    plt.xticks(n, n)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot.png')

    # plot for required p value
    plt.figure()
    P = np.arange(1, 80)
    plt.plot(P, [total(1024, adv_c, src_c, proj_c, ls_c, p) for p in P], label='N=1024')

    plt.plot(P, [16.7] * len(P), label='60 FPS', linestyle='--')
    plt.ylabel('Step time (ms)')
    plt.xlabel('Parallelization (p)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('parallel.png')


if __name__ == '__main__':
    plot()
