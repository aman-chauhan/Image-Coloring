from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import Input
from keras.models import Model


def llfn():
    # Input tensor
    llfn_input = Input(batch_shape=(None, None, None, 1), name='llfn_input')

    # Convolutional Layer with 64 3x3 kernels with double stride and same padding
    llfn_conv1 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(llfn_input)
    llfn_conv1 = BatchNormalization()(llfn_conv1)

    # Convolutional Layer with 128 3x3 kernels with single stride and same padding
    llfn_conv2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(llfn_conv1)
    llfn_conv2 = BatchNormalization()(llfn_conv2)

    # Convolutional Layer with 128 3x3 kernels with double stride and same padding
    llfn_conv3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(llfn_conv2)
    llfn_conv3 = BatchNormalization()(llfn_conv3)

    # Convolutional Layer with 256 3x3 kernels with single stride and same padding
    llfn_conv4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(llfn_conv3)
    llfn_conv4 = BatchNormalization()(llfn_conv4)

    # Convolutional Layer with 256 3x3 kernels with double stride and same padding
    llfn_conv5 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu')(llfn_conv4)
    llfn_conv5 = BatchNormalization()(llfn_conv5)

    # Convolutional Layer with 512 3x3 kernels with single stride and same padding
    llfn_conv6 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(llfn_conv5)
    llfn_conv6 = BatchNormalization()(llfn_conv6)

    # Model definition
    llfn_model = Model(inputs=llfn_input, outputs=llfn_conv6)

    return llfn_model
