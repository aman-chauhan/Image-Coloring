from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model


def clf():
    clf_input = Input(batch_shape=(None, 256), name='clf_input')
    clf_fcon1 = Dense(units=256, activation='relu')(clf_input)
    clf_fcon2 = Dense(units=300, activation='softmax')(clf_fcon1)
    clf_model = Model(inputs=clf_input, outputs=clf_fcon2, name='clf_model')
    return clf_model
