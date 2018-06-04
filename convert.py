from PIL import Image
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
                labels.update({x.split('.')[0] + '.png': '_'.join(root.split(os.sep)[8:])
                               for x in files if x.split('.')[1] == 'jpg'})
    print('Index created.')

    if not os.path.isdir(dest):
        os.mkdir(dest)
        json.dump(labels, open(os.path.join(dest, 'labels.json'), 'w'))
        print('Labels created.')

        for x in partition.keys():
            os.mkdir(os.path.join(dest, x))
            os.mkdir(os.path.join(dest, x + '-target'))
        print('Directories created.')

    for x in partition.keys():
        print('Starting ' + x)
        for i in range(len(partition[x])):
            hash = (60 * i) // len(partition[x])

            source1 = Image.open(partition[x][i])
            source1.thumbnail((224, 224), Image.LANCZOS)
            img1 = Image.new('RGB', (224, 224))
            img1.paste(source1, ((224 - source1.size[0]) // 2, (224 - source1.size[1]) // 2))

            source2 = Image.open(partition[x][i])
            source2.thumbnail((112, 112), Image.LANCZOS)
            img2 = Image.new('RGB', (112, 112))
            img2.paste(source1, ((112 - source1.size[0]) // 2, (112 - source1.size[1]) // 2))

            if not os.path.exists(os.path.join(os.path.join(dest, x + '-target'), partition[x][i].split(os.sep)[-1].split('.')[0] + '.png')):
                img2.save(os.path.join(os.path.join(dest, x + '-target'),
                                       partition[x][i].split(os.sep)[-1].split('.')[0] + '.png'))
            if not os.path.exists(os.path.join(os.path.join(dest, x), partition[x][i].split(os.sep)[-1].split('.')[0] + '.png')):
                img1.convert('L').save(os.path.join(os.path.join(dest, x),
                                                    partition[x][i].split(os.sep)[-1].split('.')[0] + '.png'))
            print('{}[{}{}]{}%'.format(x, '#' * hash, ' ' * (60 - hash), (100 * i) // len(partition[x])), end='\r')
        print('\ndone.')
    print('Dataset generated')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
