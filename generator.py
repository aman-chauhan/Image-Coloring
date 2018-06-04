from keras.utils import Sequence, to_categorical
from skimage.color import rgb2lab
from imageio import imread
import numpy as np
import os


class DataGenerator(Sequence):
    def __init__(self, list_IDs, partition, labels, batch_size=16, n_channel=1, n_classes=871, shuffle=True):
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.n_channel = n_channel
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.partition = partition
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            prng = np.random.RandomState(42)
            prng.shuffle(self.indexes)

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
            X[i] = np.expand_dims(imread(os.path.join(os.path.join('data', self.partition), ID))/255.0, axis=-1)
            Y[i] = (rgb2lab(imread(os.path.join(os.path.join('data', self.partition + '-target'), ID)))
                    [:, :, 1:] + 128.0) / (255.0)
            y[i] = self.labels[ID]
        return ([X, X], [Y, to_categorical(y, num_classes=self.n_classes)])
