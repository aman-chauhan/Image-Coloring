from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical
from skimage.color import rgb2lab
from imageio import imread
import numpy as np
import os


class DataGenerator(Sequence):
    def __init__(self, list_IDs, partition, labels, batch_size=28, n_channel=1, n_classes=719, shuffle=True, augment=False):
        if augment:
            self.list_IDs = list_IDs * 3
        else:
            self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.n_channel = n_channel
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.partition = partition
        self.prng = np.random.RandomState(42)
        self.datagen = ImageDataGenerator(rotation_range=90, width_shift_range=0.3, height_shift_range=0.3, shear_range=0.2,
                                          fill_mode='constant', cval=0, zoom_range=0.2, horizontal_flip=True, vertical_flip=True)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            self.prng.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, 224, 224, self.n_channel))
        Y = np.empty((self.batch_size, 224, 224, 2))
        y = np.empty((self.batch_size,), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            if not self.augment:
                X[i] = np.expand_dims(
                    (imread(os.path.join(os.path.join('data', self.partition), ID)) - 127.5) / 127.5, axis=-1)
                Y[i] = (rgb2lab(imread(os.path.join(os.path.join('data', self.partition + '-target'), ID)))
                        [:, :, 1:] + 128.0) / (255.0)
                y[i] = self.labels[ID]
            else:
                seed = self.prng.randint(0, 1000000)
                X[i] = (self.datagen.random_transform(np.expand_dims(
                    imread(os.path.join(os.path.join('data', self.partition), ID)), axis=-1), seed=seed) - 127.5) / 127.5
                Y[i] = (rgb2lab(self.datagen.random_transform(imread(os.path.join(os.path.join('data', self.partition + '-target'), ID)), seed=seed))
                        [:, :, 1:] + 128.0) / (255.0)
                y[i] = self.labels[ID]
        return ([X, X], [Y, to_categorical(y, num_classes=self.n_classes)])
