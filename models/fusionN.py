//redundant
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda
from keras.layers import Input
from keras.models import Model
from keras import backend as K


def tile(x):
    x = K.expand_dims(x, 1)
    x = K.expand_dims(x, 1)
    x = K.tile(x, [1, -1, -1, 1])
    return x


def fusion():
    fmlfn_input = Input(batch_shape=(None, None, None, 256), name='fusion_mlfn_input')
    fgfn_input = Input(batch_shape=(None, 256), name='fusion_gfn_input')
    fusion_layer = Concatenate()([fmlfn_input, Lambda(tile)(fgfn_input)])
    fusion_model = Model(inputs=[fmlfn_input, fgfn_input], outputs=fusion_layer, name='fusion_model')
    return fusion_model
