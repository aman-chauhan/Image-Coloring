from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model


def gfn():
    gfn_input = Input(batch_shape=(None, 28, 28, 512), name='gfn_input')
    gfn_conv1 = Conv2D(filters=512, kernel_size=3, strides=2, activation='relu')(gfn_input)
    gfn_conv2 = Conv2D(filters=512, kernel_size=3, strides=1, activation='relu')(gfn_conv1)
    gfn_conv3 = Conv2D(filters=512, kernel_size=3, strides=2, activation='relu')(gfn_conv2)
    gfn_conv4 = Conv2D(filters=512, kernel_size=3, strides=1, activation='relu')(gfn_conv3)
    gfn_flttn = Flatten()(gfn_conv4)
    gfn_fcon1 = Dense(units=1024, activation='relu')(gfn_flttn)
    gfn_fcon2 = Dense(units=512, activation='relu')(gfn_fcon1)
    gfn_fcon3 = Dense(units=256, activation='relu')(gfn_fcon2)
    gfn_model = Model(inputs=gfn_input, outputs=gfn_fcon3, name='gfn_model')
    return gfn_model
