from keras.layers.convolutional import Conv2D
from keras.layers import Input
from keras.models import Model


def llfn():
    llfn_input = Input(batch_shape=(None, None, None, 1), name='llfn_input')
    llfn_conv1 = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu')(llfn_input)
    llfn_conv2 = Conv2D(filters=128, kernel_size=3, strides=1, activation='relu')(llfn_conv1)
    llfn_conv3 = Conv2D(filters=128, kernel_size=3, strides=2, activation='relu')(llfn_conv2)
    llfn_conv4 = Conv2D(filters=256, kernel_size=3, strides=1, activation='relu')(llfn_conv3)
    llfn_conv5 = Conv2D(filters=256, kernel_size=3, strides=2, activation='relu')(llfn_conv4)
    llfn_conv6 = Conv2D(filters=512, kernel_size=3, strides=1, activation='relu')(llfn_conv5)
    llfn_model = Model(inputs=llfn_input, outputs=llfn_conv6)
    return llfn_model
