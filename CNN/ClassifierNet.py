from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras import regularizers


def clf():
    # Input tensor
    clf_input = Input(batch_shape=(None, 256), name='clf_input')

    # Fully Connected Layer with 256 units
    clf_fcon1 = Dense(units=256, activation='relu', kernel_initializer='he_uniform',
                      bias_initializer='he_uniform')(clf_input)
    clf_fcon1 = BatchNormalization()(clf_fcon1)
    clf_fcon1 = Dropout(0.20)(clf_fcon1)

    # Fully Connected Layer with 'output' units
    clf_fcon2 = Dense(units=719, activation='softmax', kernel_initializer='he_uniform',
                      bias_initializer='he_uniform')(clf_fcon1)

    # Model definition
    clf_model = Model(inputs=clf_input, outputs=clf_fcon2, name='clf_model')

    return clf_model
