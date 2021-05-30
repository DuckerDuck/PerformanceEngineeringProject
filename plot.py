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

def project_cuda(n, ls_c, proj_c, p):
    """ Analytical model for project function for second cuda model
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

def total_cuda(n, adv_c, src_c, proj_c, ls_c, cpy_c, p):
    """ CUDA analytical model """
    return 3 * add_source(n, src_c) + \
           2 * project(n, ls_c, proj_c, p) + \
           3 * lin_solve(n, ls_c, p) + \
           3 * advect(n, adv_c) + \
           6 * copy_mem(n, cpy_c)

def total_cuda_new(n, adv_c, src_c, proj_c, ls_c, cpy_c, p):
    """ CUDA analytical model v2"""
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

    # Data first parallel model + fp16
    _, y_cuda_fp16_total = parse_output(Path('./output_cuda_fp16'), 'threads', 'total step')
    _, y_cuda_fp16_linsolve = parse_output(Path('./output_cuda_fp16'), 'threads', 'lin_solve')


    # Data fully parallel model + async memory transfers
    output_cuda_new = Path('./output_cuda_new_3')
    _, y_cuda_cpy = parse_output(output_cuda_new, 'N', 'CUDA copy')
    
    _, y_cuda_new_total = parse_output(output_cuda_new, 'N', 'total step')
    _, y_cuda_new_linsolve = parse_output(output_cuda_new, 'N', 'lin_solve')
    _, y_cuda_new_advect = parse_output(output_cuda_new, 'N', 'advect')
    _, y_cuda_new_project = parse_output(output_cuda_new, 'N', 'project')
    _, y_cuda_new_source = parse_output(output_cuda_new, 'N', 'add_source')
    _, y_cuda_new_cpy_async = parse_output(output_cuda_new, 'N', 'CUDA copy async')

    # Data second parallel model + fp16
    output_cuda_new_fp16 = Path('./output_cuda_new_fp16')
    _, y_cuda_new_fp16_total = parse_output(output_cuda_new_fp16, 'N', 'total step')
    _, y_cuda_new_fp16_linsolve = parse_output(output_cuda_new_fp16, 'N', 'lin_solve')


    # Fit parameters of sequential analytical model
    ls_c = np.mean([y / (20*nn*nn) for y, nn in zip(y_linsolve, n)])
    proj_c = np.mean([(y - lin_solve(nn, ls_c, 1)) / (2 * nn * nn) for y, nn in zip(y_project, n)])
    src_c = np.mean([y / (nn*nn) for y, nn in zip(y_source, n)])
    adv_c = np.mean([y / (nn*nn) for y, nn in zip(y_advect, n)])
    print(f'Parameters Sequential: \n\tls_c: {ls_c}\n\tproj_c: {proj_c}\n\tsrc_c: {src_c}\n\tadv_c: {adv_c}')

    # Parameters of parallel model
    ls_cuda_c = np.mean([y*nn / (20*nn*nn) for y, nn in zip(y_cuda_linsolve, n)])
    ls_cuda_fp16_c = np.mean([y*p / (20*nn*nn) for p, y, nn in zip(P, y_cuda_fp16_linsolve, n)])
    cpy_cuda_c = np.mean(y_cuda_cpy)

    # ls_cuda_c benchmark includes copies, so we remove those here
    ls_cuda_c -= 2*cpy_cuda_c

    print(f'Parameters CUDA: \n\tls_cuda_c: {ls_cuda_c}\n\tls_cuda_fp16_c:{ls_cuda_fp16_c}\n\tcpy_cuda_c:{cpy_cuda_c}')

    ls_cuda_new_c = np.mean([y / 20 for y in y_cuda_new_linsolve])
    proj_cuda_new_c = np.mean([(y - ls_cuda_new_c) / 2  for y in y_cuda_new_project])
    src_cuda_new_c = np.mean(y_cuda_new_source)
    adv_cuda_new_c = np.mean(y_cuda_new_advect)
    cpy_cuda_new_c = np.mean(y_cuda_new_cpy_async)
    
    ls_cuda_new_fp16_c = np.mean([y / 20 for y in y_cuda_new_fp16_linsolve])
    print(f'Parameters CUDA v2: \n\t'
          f'ls_cuda_new_c: {ls_cuda_new_c}\n\t'
          f'ls_cuda_new_fp16_c:{ls_cuda_new_fp16_c}\n\t'
          f'proj_cuda_new_c:{proj_cuda_new_c}\n\t'
          f'src_cuda_new_c:{src_cuda_new_c}\n\t'
          f'adv_cuda_new_c:{adv_cuda_new_c}\n\t'
          f'cpy_cuda_new_c:{cpy_cuda_new_c}')


    # Speedup
    speedup = y_total[-1] / y_cuda_total[-1]
    speedup_fp16 = y_total[-1] / y_cuda_new_fp16_total[-1]
    speedup_new = y_total[-1] / y_cuda_new_total[-1]
    print(f'Speedup n={n[-1]} x{speedup}')
    print(f'Speedup v2 n={n[-1]} x{speedup_new}')
    print(f'Speedup fp16 n={n[-1]} x{speedup_fp16}')
    
    plt.figure()
    plt.scatter(n, y_total, label='Sequential')
    plt.scatter(n, y_cuda_total, label='CUDA')
    plt.scatter(n, y_cuda_new_total, label='CUDA v2')
    # plt.scatter(n, y_cuda_new_fp16_total, label='CUDA v2 FP16')
    plt.plot(n, [total(nn, adv_c, src_c, proj_c, ls_c) for nn in n], label='Sequential Model')
    plt.plot(n, [total_cuda(nn, adv_c, src_c, proj_c, ls_cuda_c, cpy_cuda_c, nn) for nn in n], label='CUDA Model')
    plt.plot(n, [total_cuda_new(nn, adv_cuda_new_c, src_cuda_new_c, proj_cuda_new_c, ls_cuda_new_c, cpy_cuda_new_c, nn*nn) for nn in n], label='CUDA Model v2')

    # Find 60 FPS gridsize limit
    for grid in range(1, 5000):
        step_time = total_cuda_new(grid, adv_cuda_new_c, src_cuda_new_c, proj_cuda_new_c, ls_cuda_new_c, cpy_cuda_new_c, grid*grid)
        if step_time > 16.7:
            print('Maximum grid size @ 60 FPS:', grid)
            break
    
    # Plot what we want
    plt.plot(n, [16.7] * len(n), label='60 FPS', linestyle='--')
    # plt.plot(n, [6.9] * len(n), label='144 FPS', linestyle='--')

    plt.ylabel('Step time (ms)')
    plt.xlabel('Grid size (N)')
    plt.ylim(0, 400)
    plt.xticks(n, n)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot.png')

    # plot for required p value
    plt.figure()
    P = np.arange(1, 1000)
    plt.plot(P, [total(1080, adv_c, src_c, proj_c, ls_cuda_c, p) for p in P], label='CUDA Model, N=1080')

    plt.plot(P, [16.7] * len(P), label='60 FPS', linestyle='--')
    plt.ylabel('Step time (ms)')
    plt.xlabel('Threads')
    plt.ylim(-100, 1000)
    plt.legend()
    plt.tight_layout()
    plt.savefig('parallel.png')

    # plot for half precision accuracy
    plt.figure()
    mse = [0.000000, 0.000001, 0.000002, 0.000004, 0.000004, 0.000006, 0.000008, 0.000013, 0.000022, 0.000035, 0.000053, 0.000076, 0.000114, 0.000141, 0.000222, 0.000244, 0.000263, 0.000279, 0.000362, 0.000465, 0.000571, 0.000819, 0.000973, 0.001256, 0.001577, 0.001916, 0.002251, 0.003209, 0.004431, 0.004775, 0.005578, 0.006325, 0.008477, 0.009638, 0.010948, 0.014514, 0.016616, 0.019748, 0.022303, 0.024529, 0.034824, 0.042314, 0.050466, 0.064348, 0.082271, 0.104473, 0.133356, 0.171123, 0.204111, 0.256029]
    steps = np.arange(1, 51)
    plt.plot(steps, mse)

    plt.ylabel('MSE')
    plt.xlabel('Steps')
    plt.tight_layout()
    plt.savefig('accuracy.png')


if __name__ == '__main__':
    plot()
