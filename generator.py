from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical
from skimage.color import rgb2lab
from imageio import imread
from PIL import Image

import numpy as np
import os


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
