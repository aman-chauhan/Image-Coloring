from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import Callback
from skimage.color import rgb2lab
from keras import backend as K
from imageio import imread
from PIL import Image

import pandas as pd
import numpy as np
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
    model_d = {'densenet': 'dn121', 'inceptionresnet': 'irv2',
               'inception': 'iv3', 'resnet': 'r50',
               'vgg': 'vgg', 'xception': 'x'}
    if key == 'densenet':
        from models.IC_DN121 import get_model
        model = get_model(model_d[key])
    elif key == 'inceptionresnet':
        from models.IC_IRV2 import get_model
        model = get_model(model_d[key])
    elif key == 'inception':
        from models.IC_IV3 import get_model
        model = get_model(model_d[key])
    elif key == 'resnet':
        from models.IC_R50 import get_model
        model = get_model(model_d[key])
    elif key == 'vgg':
        from models.IC_VGG19 import get_model
        model = get_model(model_d[key])
    elif key == 'xception':
        from models.IC_X import get_model
        model = get_model(model_d[key])
    if not os.path.exists(os.path.join('weights', '{}.h5'.format(key))):
        print('Generating new {} model.'.format(key))
    else:
        print('Fetching {} model from logs.'.format(key))
        model_path = os.path.join('weights', '{}.h5'.format(key))
        log_path = os.path.join('logs', '{}.csv'.format(key))
        model.load_weights(model_path)
        logs = pd.read_csv(log_path)
        epoch = len(logs)
        del logs
    metric = '{}_class'.format(model_d[key])
    model.compile(optimizer='adadelta',
                  loss=['mse', 'categorical_crossentropy'],
                  metrics={metric: ['categorical_accuracy',
                                    'top_k_categorical_accuracy']})
    return (model, epoch)


class ModelCheckpoint(Callback):
    def __init__(self, filepath):
        self.filepath = filepath
        self.val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] < self.val_loss:
            self.val_loss = logs.get('val_loss')
            self.model.save_weights(self.filepath)


# inspired from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(Sequence):
    def __init__(self, files, batch_size, img_size, classes, shuffle, augment):
        self.files = files
        self.batch_size = batch_size
        self.img_size = img_size
        self.classes = classes
        self.shuffle = shuffle
        self.augment = augment
        self.prng = np.random.RandomState(42)
        self.datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1,
                                          height_shift_range=0.1, fill_mode='constant',
                                          cval=0, zoom_range=0.1)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            self.prng.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_files_temp = [self.files[k] for k in indexes]
        return self.__data_generation(list_files_temp)

    def __data_generation(self, files):
        X = np.empty((self.batch_size, self.img_size, self.img_size, 1))
        Y_color = np.empty((self.batch_size, 256, 256, 2))
        Y_class = np.empty((self.batch_size,), dtype=int)
        for i, file in enumerate(files):
            file = os.path.join('places365_standard', file)
            img = imread(file)
            if self.augment:
                seed = self.prng.randint(0, 1000)
                img = self.datagen.random_transform(img, seed=seed)
            if self.img_size == 256:
                lab = rgb2lab(img)
                X[i] = np.clip(lab[:, :, 0:1] * 2.55, 0, 255)
                Y_color[i] = np.clip((lab[:, :, 1:3] + 128.0) / 255.0, 0, 1)
            else:
                timg = np.array(Image.fromarray(img).resize((self.img_size,
                                                             self.img_size),
                                                            Image.BICUBIC))
                X[i] = np.clip(rgb2lab(timg)[:, :, 0:1] * 2.55, 0, 255)
                Y_color[i] = np.clip((rgb2lab(img)[:, :, 1:3] + 128.0) / 255.0, 0, 1)
            Y_class[i] = self.classes[file.split(os.sep)[2]]
        return ([X, X], [Y_color, to_categorical(Y_class, num_classes=len(self.classes))])


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
    checkpoint = ModelCheckpoint(filepath=model_path)
    earlystop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    csvlogger = CSVLogger(filename=log_path, append=True)
    model.fit_generator(generator=train_generator, epochs=100, verbose=1,
                        callbacks=[csvlogger, checkpoint, earlystop],
                        validation_data=val_generator,
                        use_multiprocessing=True, workers=4,
                        initial_epoch=epoch)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
