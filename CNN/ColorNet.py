from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.layers import Input
from keras.models import Model


def color():
    # Input tensor
    color_input = Input(batch_shape=(None, None, None, 512), name='color_input')

    # Convolutional Layer with 256 3x3 kernels with single stride and same padding
    color_conv1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(color_input)
    # color_conv1 = BatchNormalization()(color_conv1)

    # Convolutional Layer with 128 3x3 kernels with single stride and same padding
    color_conv2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(color_conv1)
    # color_conv2 = BatchNormalization()(color_conv2)

    # Upsampling
    color_upsm1 = UpSampling2D(size=2)(color_conv2)

    # Convolutional Layer with 64 3x3 kernels with single stride and same padding
    color_conv3 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(color_upsm1)
    # color_conv3 = BatchNormalization()(color_conv3)

    # Convolutional Layer with 64 3x3 kernels with single stride and same padding
    color_conv4 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(color_conv3)
    # color_conv4 = BatchNormalization()(color_conv4)

    # Upsampling
    color_upsm2 = UpSampling2D(size=2)(color_conv4)

    # Convolutional Layer with 32 3x3 kernels with single stride and same padding
    color_conv5 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(color_upsm2)
    # color_conv5 = BatchNormalization()(color_conv5)

    # Convolutional Layer with 2 3x3 kernels with single stride and same padding
    color_conv6 = Conv2D(filters=2, kernel_size=3, strides=1, padding='same',
                         activation='sigmoid', name='color_output')(color_conv5)

    # Model definition
    color_model = Model(inputs=color_input, outputs=color_conv6, name='color_model')

    return color_model
