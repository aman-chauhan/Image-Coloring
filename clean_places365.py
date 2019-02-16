from imageio import imread
import numpy as np
import sys
import os


def main(filename):
    filepath = os.path.join('places365_standard', '{}.txt'.format(filename))
    files = None
    with open(filepath, 'r') as fp:
        files = [x.strip() for x in fp.readlines()]

    with open('{}.txt'.format(filename), 'w') as fp:
        for i, file in enumerate(files):
            img = imread(file)
            if len(img.shape) == 2 or img.shape[2] == 1:
                print('Skipping file - {}'.format(file))
                del img
                continue
            del img
            fp.write('{}\n'.format(file))
            if i % 100 == 0:
                print('.', end='')
            if i % 10000 == 0:
                print()


if __name__ == '__main__':
    main(sys.argv[1])
