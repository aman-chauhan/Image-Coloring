from keras.utils import multi_gpu_model
from keras.callbacks import Callback
from generator import DataGenerator
from CNN import FullNetwork
import tensorflow as tf
import json
import sys
import os


class SaveCallback(Callback):
    def __init__(self, model):
        self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save_weights('weights.h5')
        d = {}
        if os.path.exists('epochs.json'):
            d = json.load(open('epochs.json'))
        d[epoch] = logs
        json.dump(d, open('epochs.json', 'w'))


def main():
    labels = json.load(open(os.path.join('data', 'labels.json')))
    partition = {'training': None, 'validation': None}
    for x in partition.keys():
        partition[x] = [f for f in os.listdir(os.path.join('data', x)) if os.path.isfile(
            os.path.join(os.path.join('data', x), f))]
        partition[x].sort()
    print('Indices read.')

    n_classes = len({labels[x] for x in labels})
    l = {labels[x] for x in labels}
    l = {x: i for i, x in enumerate(sorted(list(l)))}
    labels = {x: l[labels[x]] for x in labels.keys()}
    json.dump(l, open('mapping.json', 'w'))
    print('Mappings written.')

    training_generator = DataGenerator(partition['training'], 'training', labels, 28, 1, n_classes, True, True)
    validation_generator = DataGenerator(partition['validation'], 'validation', labels, 28, 1, n_classes, True, True)

    model = None
    with tf.device('/cpu:0'):
        model = FullNetwork.model()
    if os.path.exists('weights.h5'):
        model.load_weights('weights.h5')

    initial_epoch = 0
    if os.path.exists('epochs.json'):
        initial_epoch = len(json.load(open('epochs.json')).keys())

    cbk = SaveCallback(model)
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(optimizer='adam', loss={
        'color_model': 'mean_squared_error', 'clf_model': 'categorical_crossentropy'}, metrics={'color_model': 'accuracy', 'clf_model': 'accuracy'})
    parallel_model.fit_generator(generator=training_generator, epochs=1000, verbose=2, callbacks=[
        cbk], validation_data=validation_generator, use_multiprocessing=True, workers=4, initial_epoch=initial_epoch)
    print('Training done.')


if __name__ == '__main__':
    main()
