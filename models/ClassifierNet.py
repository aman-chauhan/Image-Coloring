from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model


def clf():
    # Input tensor
    clf_input = Input(batch_shape=(None, 512), name='clf_input')

    # Fully Connected Layer with 256 units
    clf_fcon1 = Dense(units=256, activation='relu')(clf_input)
    clf_fcon1 = BatchNormalization()(clf_fcon1)

    # Fully Connected Layer with 'output' units
    clf_fcon2 = Dense(units=300, activation='softmax')(clf_fcon1)

    # Model definition
    clf_model = Model(inputs=clf_input, outputs=clf_fcon2, name='clf_model')

    return clf_model
