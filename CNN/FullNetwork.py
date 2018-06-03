from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras import backend as K

from .LowLevelFeatureNet import llfn
from .MidLevelFeatureNet import mlfn
from .ClassifierNet import clf
from .GlobalFeatureNet import gfn
from .ColorNet import color


def tile(x, k):
    x = K.expand_dims(x, 1)
    x = K.expand_dims(x, 1)
    x = K.tile(x, [1, k[1], k[2], 1])
    return x


def model():
    color_input = Input(batch_shape=(None, None, None, 1), name='global_color')
    color_branch = llfn()(color_input)
    color_branch = mlfn()(color_branch)

    class_input = Input(batch_shape=(None, 224, 224, 1), name='global_class')
    class_branch = llfn()(class_input)
    class_branch = gfn()(class_branch)

    gfn_units = Dense(units=256, activation='relu')(class_branch)
    gfn_units = BatchNormalization()(gfn_units)

    color_branch = Concatenate()([color_branch, Lambda(tile,arguments={'k':K.shape(color_branch)})(gfn_units)])
    color_branch = color()(color_branch)

    class_branch = clf()(class_branch)

    model = Model(inputs=[color_input, class_input], outputs=[color_branch, class_branch], name='global_model')
    return model


if __name__ == '__main__':
    print(model().summary())
