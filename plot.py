import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from numpy.lib.function_base import copy


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

def project_cuda(n, ls_c, proj_c, p):
    """ Analytical model for project function
    n: grid size
    ls_c: lin solve kernel constant
    proj_c: project kernel constant
    """
    return lin_solve(n, ls_c, p) + (2 * n * n * proj_c) / p

def add_source(n, src_c, p=1):
    """ Analytical model for add_source function
    n: grid size
    src_c: add_source kernel constant
    """
    return (n*n*src_c) / p

def copy_mem(n, cpy_c):
    return n * n * cpy_c

def advect(n, adv_c, p=1):
    """ Analytical model for advect function
    n: grid size
    adv_c: advect kernel constant
    """
    return n*n*adv_c / p

def total(n, adv_c, src_c, proj_c, ls_c, p=1):
    """ Sequential analytical model """
    return 3 * add_source(n, src_c) + \
           2 * project(n, ls_c, proj_c, p) + \
           3 * lin_solve(n, ls_c, p) + \
           3 * advect(n, adv_c)

def total_cuda_1(n, adv_c, src_c, proj_c, ls_c, cpy_c, p):
    """ CUDA analytical model 1 """
    return 3 * add_source(n, src_c) + \
           2 * project(n, ls_c, proj_c, p) + \
           3 * lin_solve(n, ls_c, p) + \
           3 * advect(n, adv_c) + \
           5 * copy_mem(n, cpy_c)

def total_cuda_new(n, adv_c, src_c, proj_c, ls_c, cpy_c, p):
    """ CUDA analytical model 2 """
    return 3 * add_source(n, src_c, p) + \
           2 * project_cuda(n, ls_c, proj_c, p) + \
           3 * lin_solve(n, ls_c, p) + \
           3 * advect(n, adv_c, p) + \
           6 * copy_mem(n, cpy_c)

def plot():
    plt.rcParams.update({'font.size': 12})
    
    output = Path('./output')
    n, y_total = parse_output(output, 'N', 'total step')
    _, y_advect = parse_output(output, 'N', 'advect')
    _, y_linsolve = parse_output(output, 'N', 'lin_solve')
    _, y_project = parse_output(output, 'N', 'project')
    _, y_source = parse_output(output, 'N', 'add_source')
    
    # Data first parallel model
    output_cuda = Path('./output_cuda')
    P, y_cuda_total = parse_output(output_cuda, 'threads', 'total step')
    _, y_cuda_linsolve = parse_output(output_cuda, 'threads', 'lin_solve')

    # Data first parallel model  + fp16
    _, y_cuda_fp16_total = parse_output(Path('./output_cuda_fp16'), 'threads', 'total step')
    _, y_cuda_fp16_linsolve = parse_output(Path('./output_cuda_fp16'), 'threads', 'lin_solve')

    # Data fully parallel model + async memory transfers
    output_cuda_new = Path('./output_cuda_new')
    _, y_cuda_new_total = parse_output(output_cuda_new, 'N', 'total step')
    _, y_cuda_new_linsolve = parse_output(output_cuda_new, 'N', 'lin_solve')
    _, y_cuda_new_advect = parse_output(output_cuda_new, 'N', 'advect')
    _, y_cuda_new_project = parse_output(output_cuda_new, 'N', 'project')
    _, y_cuda_new_project = parse_output(output_cuda_new, 'N', 'add_source')


    # Fit parameters of sequential analytical model
    ls_c = np.mean([y / (20*nn*nn) for y, nn in zip(y_linsolve, n)])
    proj_c = np.mean([(y - lin_solve(nn, ls_c, 1)) / (2 * nn * nn) for y, nn in zip(y_project, n)])
    src_c = np.mean([y / (nn*nn) for y, nn in zip(y_source, n)])
    adv_c = np.mean([y / (nn*nn) for y, nn in zip(y_advect, n)])
    print(f'Parameters Sequential: \n\tls_c: {ls_c}\n\tproj_c: {proj_c}\n\tsrc_c: {src_c}\n\tadv_c: {adv_c}')

    # Parameters of parallel model
    ls_cuda_c = np.mean([y*nn / (20*nn*nn) for y, nn in zip(y_cuda_linsolve, n)])
    ls_cuda_fp16_c = np.mean([y*p / (20*nn*nn) for p, y, nn in zip(P, y_cuda_fp16_linsolve, n)])
    cpy_cuda_c = np.mean([0.0000172542, 0.0000131765, 0.0000106673, 0.0000090736, 0.0000080918, 0.0000074619, 0.0000070025, 0.0000079454, 0.0000082313])

    # ls_cuda_c benchmark includes copies, so we remove those here
    ls_cuda_c -= 2*cpy_cuda_c

    print(f'Parameters CUDA: \n\tls_cuda_c: {ls_cuda_c}\n\tls_cuda_fp16_c:{ls_cuda_fp16_c}\n\tcpy_cuda_c:{cpy_cuda_c}')

    # Speedup
    speedup = y_total[-1] / y_cuda_total[-1]
    speedup_fp16 = y_total[-1] / y_cuda_fp16_total[-1]
    print(f'Speedup n={n[-1]} x{speedup}')
    print(f'Speedup fp16 n={n[-1]} x{speedup_fp16}')
    
    plt.figure()
    plt.scatter(n, y_total, label='Sequential')
    plt.scatter(n, y_cuda_total, label='CUDA')
    # plt.scatter(n, y_cuda_fp16_total, label='CUDA FP16')
    plt.plot(n, [total(nn, adv_c, src_c, proj_c, ls_c) for nn in n], label='Sequential Model')
    plt.plot(n, [total_cuda_1(nn, adv_c, src_c, proj_c, ls_cuda_c, cpy_cuda_c, nn) for nn in n], label='CUDA Model')
    # plt.plot(n, [total(nn, adv_c, src_c, proj_c, ls_cuda_fp16_c, p) for nn, p in zip(n, P)], label='CUDA FP16 Model')
    
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
    P = np.arange(1, 100)
    plt.plot(P, [total(1080, adv_c, src_c, proj_c, ls_cuda_c, p) for p in P], label='CUDA Model, N=1080')

    plt.plot(P, [16.7] * len(P), label='60 FPS', linestyle='--')
    plt.ylabel('Step time (ms)')
    plt.xlabel('Threads')
    plt.legend()
    plt.tight_layout()
    plt.savefig('parallel.png')


if __name__ == '__main__':
    plot()
