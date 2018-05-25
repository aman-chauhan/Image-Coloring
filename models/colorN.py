from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.layers import Input
from keras.models import Model


def color():
    color_input = Input(batch_shape=(None, None, None, 512), name='color_input')
    color_conv1 = Conv2D(filters=256, kernel_size=3, strides=1, activation='relu')(color_input)
    color_conv2 = Conv2D(filters=128, kernel_size=3, strides=1, activation='relu')(color_conv1)
    color_upsm1 = UpSampling2D(size=2)(color_conv2)
    color_conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(color_upsm1)
    color_conv4 = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(color_conv3)
    color_upsm2 = UpSampling2D(size=2)(color_conv4)
    color_conv5 = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(color_upsm2)
    color_conv6 = Conv2D(filters=2, kernel_size=3, strides=1, activation='sigmoid')(color_conv5)
    color_model = Model(inputs=color_input, outputs=color_conv6, name='color_model')
    return color_model
