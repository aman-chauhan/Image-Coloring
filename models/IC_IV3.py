from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import Input

from keras.models import Model
from keras import backend as K


class IC_InceptionV3:
    def get_inception(self):
        inception = InceptionV3(weights='imagenet', include_top=False,
                                input_shape=(None, None, 3))
        i = 0
        while(True):
            if inception.layers[i].name == 'mixed2':
                break
            i += 1
        model = Model(inputs=inception.layers[0].input,
                      outputs=inception.layers[i].output,
                      name='inception')
        model.trainable = False
        model.compile('adadelta', 'mse')
        return model

    def get_dense_block(self, prev_filters, id):
        input_layer = Input(batch_shape=(None, None, None, prev_filters),
                            name='{}_{}_input'.format(self.dense_name, id))
        for i in range(1, 8):
            conv = Conv2D(filters=128, kernel_size=1, strides=1, padding='same',
                          kernel_initializer='he_uniform',
                          bias_initializer='he_uniform',
                          name='{}_{}_conv_{}a'.format(self.dense_name,
                                                       id, i))
            conv1 = conv(input_layer if i == 1 else layer)
            norm1 = BatchNormalization(scale=False,
                                       name='{}_{}_norm_{}a'.format(self.dense_name,
                                                                    id, i))(conv1)
            relu1 = Activation(activation='relu',
                               name='{}_{}_relu_{}a'.format(self.dense_name,
                                                            id, i))(norm1)
            conv2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same',
                           kernel_initializer='he_uniform',
                           bias_initializer='he_uniform',
                           name='{}_{}_conv_{}b'.format(self.dense_name,
                                                        id, i))(relu1)
            norm2 = BatchNormalization(scale=False,
                                       name='{}_{}_norm_{}b'.format(self.dense_name,
                                                                    id, i))(conv2)
            relu2 = Activation(activation='relu',
                               name='{}_{}_relu_{}b'.format(self.dense_name,
                                                            id, i))(norm2)
            concat = Concatenate(name='{}_{}_concat_{}'.format(self.dense_name,
                                                               id, i))
            layer = concat([input_layer if i == 1 else layer, relu2])
        return Model(inputs=input_layer,
                     outputs=layer,
                     name='{}_{}'.format(self.dense_name, id))

    def get_low_level_features(self):
        input_layer = Input(batch_shape=(None, None, None, 1),
                            name='{}_input'.format(self.low_name))
        duplicate = Lambda(lambda x: K.tile(x, [1, K.shape(x)[1], K.shape(x)[2], 3]),
                           name='{}_copy'.format(self.low_name))(input_layer)
        preprocess = Lambda(lambda x: preprocess_input(x),
                            name='{}_pre'.format(self.low_name))(duplicate)
        inception = self.get_inception()(preprocess)
        new_layers = self.get_dense_block(K.int_shape(inception)[-1], 0)(inception)
        return Model(inputs=input_layer,
                     outputs=new_layers,
                     name='{}'.format(self.low_name))

    def __init__(self, name):
        self.low_name = '{}_low'.format(name)
        self.dense_name = '{}_dense'.format(name)
        self.feed_size = 278
