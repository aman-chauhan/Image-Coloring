from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras import regularizers


def gfn():
    # Input tensor
    gfn_input = Input(batch_shape=(None, 28, 28, 256), name='gfn_input')

    # Convolutional Layer with 256 3x3 kernels with double stride and same padding
    gfn_conv1 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu',
                       kernel_initializer='he_uniform', bias_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005))(gfn_input)
    gfn_conv1 = BatchNormalization()(gfn_conv1)

    # Convolutional Layer with 256 3x3 kernels with single stride and same padding
    gfn_conv2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                       kernel_initializer='he_uniform', bias_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005))(gfn_conv1)
    gfn_conv2 = BatchNormalization()(gfn_conv2)

    # Convolutional Layer with 256 3x3 kernels with single stride and same padding
    gfn_conv3 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu',
                       kernel_initializer='he_uniform', bias_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005))(gfn_conv2)
    gfn_conv3 = BatchNormalization()(gfn_conv3)

    # Convolutional Layer with 256 3x3 kernels with single stride and same padding
    gfn_conv4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                       kernel_initializer='he_uniform', bias_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005))(gfn_conv3)
    gfn_conv4 = BatchNormalization()(gfn_conv4)

    # Flatten the layer
    gfn_flttn = Flatten()(gfn_conv4)

    # Fully Connected Layer with 1024 units
    gfn_fcon1 = Dense(units=1024, activation='relu', kernel_initializer='he_uniform',
                      bias_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005))(gfn_flttn)
    gfn_fcon1 = BatchNormalization()(gfn_fcon1)
    gfn_fcon1 = Dropout(0.20)(gfn_fcon1)

    # Fully Connected Layer with 512 units
    gfn_fcon2 = Dense(units=512, activation='relu', kernel_initializer='he_uniform',
                      bias_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005))(gfn_fcon1)
    gfn_fcon2 = BatchNormalization()(gfn_fcon2)
    gfn_fcon2 = Dropout(0.20)(gfn_fcon2)

    # Model definition
    gfn_model = Model(inputs=gfn_input, outputs=gfn_fcon2, name='gfn_model')

    return gfn_model
