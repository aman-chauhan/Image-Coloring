from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from generator import DataGenerator
from keras.models import load_model
from keras import backend as K

import pandas as pd
import models
import sys
import os


def get_classes(filename):
    lines = None
    with open(filename, 'r') as fp:
        lines = [x.strip() for x in fp.readlines()]
    return {lines[i]: i for i in range(len(lines))}


def get_filepaths(filename):
    lines = None
    with open(filename, 'r') as fp:
        lines = [x.strip() for x in fp.readlines()]
    return lines


def get_model_and_epochs(key):
    model = None
    epoch = 0
    if not os.path.exists(os.path.join('weights', '{}.h5'.format(key))):
        print('Generating new {} model.'.format(key))
        model_d = {'densenet': 'dn121', 'inceptionresnet': 'irv2',
                   'inception': 'iv3', 'resnet': 'r50',
                   'vgg': 'vgg', 'xception': 'x'}
        if key == 'densenet':
            model = models.IC_DenseNet121(model_d[key]).get_model()
        elif key == 'inceptionresnet':
            model = models.IC_InceptionResNetV2(model_d[key]).get_model()
        elif key == 'inception':
            model = models.IC_InceptionV3(model_d[key]).get_model()
        elif key == 'resnet':
            model = models.IC_ResNet50(model_d[key]).get_model()
        elif key == 'vgg':
            model = models.IC_VGG19(model_d[key]).get_model()
        elif key == 'xception':
            model = models.IC_Xception(model_d[key]).get_model()
        metric = '{}_class'.format(model_d[key])
        model.compile(optimizer='adadelta',
                      loss=['mse', 'categorical_crossentropy'],
                      metrics={metric: ['categorical_accuracy',
                                        'top_k_categorical_accuracy']})
    else:
        print('Fetching {} model from logs.'.format(key))
        model_path = os.path.join('weights', '{}.h5'.format(key))
        log_path = os.path.join('logs', '{}.csv'.format(key))
        model = load_model(model_path)
        logs = pd.read_csv(log_path)
        epoch = len(logs)
        del logs
    return (model, epoch)


def main(key, batch_size):
    K.clear_session()

    classes = get_classes('classes.txt')
    train = get_filepaths('train.txt')
    val = get_filepaths('val.txt')

    model, epoch = get_model_and_epochs(key)
    img_size = 278 if key.startswith('inception') else 256
    train_generator = DataGenerator(train, batch_size, img_size, classes, True, False)
    val_generator = DataGenerator(val, batch_size, img_size, classes, True, False)

    model_path = os.path.join('weights', '{}.h5'.format(key))
    log_path = os.path.join('logs', '{}.csv'.format(key))
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min')
    earlystop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    csvlogger = CSVLogger(filename=log_path, append=True)
    model.fit_generator(generator=train_generator, epochs=100, verbose=1,
                        callbacks=[csvlogger, checkpoint, earlystop],
                        validation_data=val_generator,
                        use_multiprocessing=True, workers=4,
                        initial_epoch=epoch)



if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
