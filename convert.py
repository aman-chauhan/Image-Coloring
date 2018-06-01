from skimage.transform import resize
from imageio import imread, imwrite
from skimage import img_as_ubyte
from skimage import color
from pprint import pprint
import numpy as np
import json
import sys
import os


def main(source, dest):
    partition = {}
    labels = {}
    for root, subdirs, files in os.walk(source):
        if len(root.split(os.sep)) > 8 and len([x for x in files if x.split('.')[1] == 'jpg']) > 0:
            if root.split(os.sep)[7] not in ['misc', 'outliers']:
                if root.split(os.sep)[6] not in partition:
                    partition[root.split(os.sep)[6]] = []
                partition[root.split(os.sep)[6]].extend([os.path.join(root, x)
                                                         for x in files if x.split('.')[1] == 'jpg'])
                labels.update({x: '_'.join(root.split(os.sep)[8:]) for x in files if x.split('.')[1] == 'jpg'})
    print('Index created.')

    if not os.path.isdir(dest):
        os.mkdir(dest)
        json.dump(labels, open(os.path.join(dest, 'labels.json'), 'w'))
        print('Labels created.')

        for x in partition.keys():
            os.mkdir(os.path.join(dest, x))
            os.mkdir(os.path.join(dest, 'target-' + x))
            os.mkdir(os.path.join(dest, 'class-' + x))
        print('Directories created.')

    for x in partition.keys():
        print('Starting ' + x)
        for i in range(len(partition[x])):
            hash = (60 * i) // len(partition[x])
            img = imread(partition[x][i])
            if not os.path.exists(os.path.join(os.path.join(dest, x), partition[x][i].split(os.sep)[-1])):
                imwrite(os.path.join(os.path.join(dest, x), partition[x][i].split(
                    os.sep)[-1]), img_as_ubyte(color.rgb2gray(img)))
            if not os.path.exists(os.path.join(os.path.join(dest, 'target-' + x), partition[x][i].split(os.sep)[-1])):
                imwrite(os.path.join(os.path.join(dest, 'target-' + x),
                                     partition[x][i].split(os.sep)[-1]), img)
            if not os.path.exists(os.path.join(os.path.join(dest, 'class-' + x), partition[x][i].split(os.sep)[-1])):
                imwrite(os.path.join(os.path.join(dest, 'class-' + x),
                                     partition[x][i].split(os.sep)[-1]), img_as_ubyte(resize(img, (224, 224, 3))))
            print('{}[{}{}]{}%'.format(x, '#' * hash, ' ' * (60 - hash), (100 * i) // len(partition[x])), end='\r')
        print('done.')
    print('Dataset generated')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
