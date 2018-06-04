from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import Input
from keras.models import Model


def mlfn():
    # Input tensor
    mlfn_input = Input(batch_shape=(None, None, None, 512), name='mlfn_input')

    # Convolutional Layer with 512 3x3 kernels with single stride and same padding
    mlfn_conv1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(mlfn_input)
    # mlfn_conv1 = BatchNormalization()(mlfn_conv1)

    # Convolutional Layer with 256 3x3 kernels with single stride and same padding
    mlfn_conv2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(mlfn_conv1)
    # mlfn_conv2 = BatchNormalization()(mlfn_conv2)

    # Model definition
    mlfn_model = Model(inputs=mlfn_input, outputs=mlfn_conv2, name='mlfn_model')

    return mlfn_model
