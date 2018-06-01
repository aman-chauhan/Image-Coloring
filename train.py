from generator import DataGenerator
import models
import json
import sys
import os


def main():
    labels = json.load(open(os.path.join('data', 'labels.json')))
    partition = {'training': None, 'validation': None}
    for x in partition.keys():
        partition[x] = [f for f in os.listdir(os.path.join('data', x)) if os.path.isfile(
            os.path.join(os.path.join('data', x), f))]
        partition[x].sort()
    print('Indices read.')

    n_classes = len({labels[x] for x in labels})
    training_generator = DataGenerator(partition['training'], 'training', labels, 16, 1, n_classes, True)
    validation_generator = DataGenerator(partition['validation'], 'validation', labels, 16, 1, n_classes, True)

    model = FullNetwork.model()
    model.compile(optimizer='adadelta', loss={
                  'color_outout': 'mean_squared_error', 'clf_output': 'binary_crossentropy'})


if __name__ == '__main__':
    main()
