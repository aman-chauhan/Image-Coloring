from keras.layers.convolutional import Conv2D
from keras.layers import Input
from keras.models import Model


def mlfn():
    mlfn_input = Input(batch_shape=(None, None, None, 512), name='mlfn_input')
    mlfn_conv1 = Conv2D(filters=512, kernel_size=3, strides=1, activation='relu')(mlfn_input)
    mlfn_conv2 = Conv2D(filters=256, kernel_size=3, strides=1, activation='relu')(mlfn_conv1)
    mlfn_model = Model(inputs=mlfn_input, outputs=mlfn_conv2, name='mlfn_model')
    return mlfn_model
