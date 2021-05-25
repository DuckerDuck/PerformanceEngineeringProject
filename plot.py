import matplotlib.pyplot as plt
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


def plot():
    x, y_total = parse_output(Path('./output'), 'N', 'total step')
    _, y_advect = parse_output(Path('./output'), 'N', 'advect')
    _, y_linsolve = parse_output(Path('./output'), 'N', 'lin_solve')
    _, y_project = parse_output(Path('./output'), 'N', 'project')
    _, y_source = parse_output(Path('./output'), 'N', 'add_source')
    
    plt.figure()
    # plt.plot(x, y_total, label='total')
    plt.plot(x, y_advect, label='advect')
    # plt.plot(x, y_linsolve, label='lin_solve')
    # plt.plot(x, y_project, label='project')
    plt.plot(x, y_source, label='add_source')
    
    plt.plot(x, [a - b for a, b in zip (y_project,y_linsolve)], label='project overhead')

    # Plot what we want
    plt.plot(x, [16.7] * len(x), label='60 FPS', linestyle='--')
    plt.plot(x, [6.9] * len(x), label='144 FPS', linestyle='--')

    plt.ylabel('Step time (ms)')
    plt.xlabel('Grid size (N)')
    plt.xticks(x, x)
    plt.legend()
    plt.savefig('plot.png')


if __name__ == '__main__':
    plot()
