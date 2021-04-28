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
    x, y = parse_output(Path('./output'), 'N', 'frames per second')
    
    plt.figure()
    plt.plot(x, y) 
    plt.ylabel('Frames per second')
    plt.xlabel('Grid size')
    plt.xticks(x, x)
    plt.savefig('plot.png')


if __name__ == '__main__':
    plot()
