from skimage.color import rgb2lab, lab2rgb
from skimage.filters import gaussian
from imageio import imread, imwrite
from keras import backend as K
from PIL import Image

import pandas as pd
import numpy as np
import shutil
import models
import json
import sys
import os


def init(root, key):
    temp_path = os.path.join(root, 'temp')
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    gray_path = os.path.join(root, 'grayscale')
    if not os.path.exists(gray_path):
        os.mkdir(gray_path)

    map_path = os.path.join(root, 'map')
    if not os.path.exists(map_path):
        os.mkdir(map_path)
    map_path = os.path.join(map_path, key)
    if not os.path.exists(map_path):
        os.mkdir(map_path)

    color_path = os.path.join(root, 'color')
    if not os.path.exists(color_path):
        os.mkdir(color_path)
    color_path = os.path.join(color_path, key)
    if not os.path.exists(color_path):
        os.mkdir(color_path)

    class_path = os.path.join(root, 'class')
    if not os.path.exists(class_path):
        os.mkdir(class_path)
    class_path = os.path.join(class_path, key)
    if not os.path.exists(class_path):
        os.mkdir(class_path)

    return (temp_path, gray_path, map_path, color_path, class_path)


def get_classes(filename):
    lines = None
    with open(filename, 'r') as fp:
        lines = [x.strip() for x in fp.readlines()]
    return {i: lines[i] for i in range(len(lines))}


def get_truth_images(root, img_size):
    path = os.path.join(root, 'truth')
    imgs = None
    files = None
    for _, _, files in os.walk(path):
        files = [x for x in files if not x.startswith('.')]
        imgs = np.empty((len(files), img_size, img_size, 1))
        for i, file in enumerate(files):
            filepath = os.path.join(path, file)
            img = imread(filepath)
            if img.shape[0] != img_size:
                img = np.array(Image.fromarray(img).resize((img_size, img_size),
                                                           Image.BICUBIC))
            imgs[i] = rgb2lab(img)[:, :, 0:1] * 2.55
    return (imgs, files)


def get_model_and_size(key):
    model = None
    img_size = 278 if key.startswith('inception') else 256
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
    model_path = os.path.join('weights', '{}.h5'.format(key))
    model.load_weights(model_path)
    return (model, img_size)


def single(root, key):
    temp_path, gray_path, map_path, color_path, class_path = init(root, key)
    true_path = os.path.join(root, 'truth')
    classes = get_classes('classes.txt')
    model, img_size = get_model_and_size(key)
    batch, filenames = get_truth_images(root, img_size)

    imgs, class_pred = model.predict([batch, batch])
    np.save(os.path.join(temp_path, '{}_imgs'.format(key)), imgs)
    np.save(os.path.join(temp_path, '{}_pred'.format(key)), class_pred)
    class_args = np.argsort(class_pred, axis=1)[:, -5:][:, ::-1]

    for i, file in enumerate(filenames):
        gray_img = rgb2lab(imread(os.path.join(true_path, file)))[:, :, 0:1]
        const_img = gaussian(np.random.rand(*gray_img.shape) * 50.0 + 25.0)
        # color image
        img = np.concatenate((gray_img,
                              np.clip(imgs[i] * 255.0 - 128.0, -128.0, 127.0)),
                             axis=-1)
        img = lab2rgb(img) * 255.0
        imwrite(os.path.join(color_path, file),
                np.clip(img.astype('uint8'), 0, 255))
        # map image
        img = np.concatenate((const_img,
                              np.clip(imgs[i] * 255.0 - 128.0, -128.0, 127.0)),
                             axis=-1)
        img = lab2rgb(img) * 255.0
        imwrite(os.path.join(map_path, file),
                np.clip(img.astype('uint8'), 0, 255))
        # class prediction
        d = {classes[x]: float(class_pred[i, x]) * 100.0 for x in class_args[i]}
        json.dump(d, open(os.path.join(class_path,
                                       '{}.json'.format(file.split('.')[0])),
                          'w'),
                  indent=4, sort_keys=True)
        # grayscale image
        if not os.path.exists(os.path.join(gray_path, file)):
            imwrite(os.path.join(gray_path, file),
                    (gray_img[:, :, 0] * 2.55).astype('uint8'))


def all(root, key):
    temp_path, gray_path, map_path, color_path, class_path = init(root, key)
    true_path = os.path.join(root, 'truth')
    classes = get_classes('classes.txt')

    model_names = None
    for _, _, files in os.walk('weights'):
        model_names = [x.strip().split('.')[0] for x in files if not x.startswith('.')]

    _, filenames = get_truth_images(root, 256)

    imgs = None
    class_pred = None
    for model_name in model_names:
        if imgs is None:
            imgs = np.load(os.path.join(temp_path, '{}_imgs.npy'.format(model_name)))
            class_pred = np.load(os.path.join(temp_path, '{}_pred.npy'.format(model_name)))
        else:
            imgs += np.load(os.path.join(temp_path, '{}_imgs.npy'.format(model_name)))
            class_pred += np.load(os.path.join(temp_path, '{}_pred.npy'.format(model_name)))
    imgs /= len(model_names)
    class_pred /= len(model_names)
    class_args = np.argsort(class_pred, axis=1)[:, -5:][:, ::-1]

    shutil.rmtree(temp_path)

    for i, file in enumerate(filenames):
        gray_img = rgb2lab(imread(os.path.join(true_path, file)))[:, :, 0:1]
        const_img = gaussian(np.random.rand(*gray_img.shape) * 50.0 + 25.0)
        # color image
        img = np.concatenate((gray_img,
                              np.clip(imgs[i] * 255.0 - 128.0, -128.0, 127.0)),
                             axis=-1)
        img = lab2rgb(img) * 255.0
        imwrite(os.path.join(color_path, file),
                np.clip(img.astype('uint8'), 0, 255))
        # map image
        img = np.concatenate((const_img,
                              np.clip(imgs[i] * 255.0 - 128.0, -128.0, 127.0)),
                             axis=-1)
        img = lab2rgb(img) * 255.0
        imwrite(os.path.join(map_path, file),
                np.clip(img.astype('uint8'), 0, 255))
        # class prediction
        d = {classes[x]: float(class_pred[i, x]) * 100.0 for x in class_args[i]}
        json.dump(d, open(os.path.join(class_path,
                                       '{}.json'.format(file.split('.')[0])),
                          'w'),
                  indent=4, sort_keys=True)
        # grayscale image
        if not os.path.exists(os.path.join(gray_path, file)):
            imwrite(os.path.join(gray_path, file),
                    (gray_img[:, :, 0] * 2.55).astype('uint8'))


if __name__ == '__main__':
    key = sys.argv[2]
    if key == 'all':
        all(sys.argv[1], key)
    else:
        single(sys.argv[1], key)
