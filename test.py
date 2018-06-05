from skimage.color import rgb2lab, lab2rgb
from imageio import imread, imwrite
from CNN import FullNetwork
from PIL import Image

import tensorflow as tf
import numpy as np
import json
import sys
import os


def main(filename):
    d = json.load(open('mappings.json'))
    d = {v: k for k, v in d.items()}
    print('Mappings read.')

    Image.open(filename).convert('L').save(filename.split('.')[0] + '_input.png')
    source1 = Image.open(filename).convert('L')
    source1.thumbnail((224, 224), Image.LANCZOS)
    img1 = Image.new('L', (224, 224))
    img1.paste(source1, ((224 - source1.size[0]) // 2, (224 - source1.size[1]) // 2))
    clf = np.array(img1)
    clf = np.expand_dims(np.expand_dims(clf, -1), 0)
    img = np.expand_dims(np.expand_dims(rgb2lab(imread(filename))[:, :, 0], -1), 0) / 255.0

    model = FullNetwork.model()
    if os.path.exists('weights.h5'):
        model.load_weights('weights.h5')
        print('Model loaded.')

    ab, pred = model.predict([img, clf], 1, 1,)
    ab = np.clip((ab * 255.0) - 128.0, -128.0, 127.0)
    img = rgb2lab(imread(filename))
    if ab[0].shape != img[:, :, 1:].shape:
        row_diff = (ab[0].shape[0] - img[:, :, 1:].shape[0]) // 2
        col_diff = (ab[0].shape[1] - img[:, :, 1:].shape[1]) // 2
        img[:, :, 1:] = ab[0, row_diff:ab[0].shape[0] - row_diff, col_diff:ab[0].shape[1] - col_diff, :]
    else:
        img[:, :, 1:] = ab[0]
    imwrite(filename.split('.')[0] + '_output.png', lab2rgb(img))
    v = np.nonzero(pred)[1]
    p = {d[i]: float(pred[0, i]) for i in v}
    json.dump(p, open(filename.split('.')[0] + '_prediction.json', 'w'))
    print('Generation done.')


if __name__ == '__main__':
    main(sys.argv[1])
