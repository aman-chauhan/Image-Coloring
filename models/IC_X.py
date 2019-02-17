from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input

from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import Input

from keras.models import Model
from keras import backend as K


class IC_Xception:
    def get_xception(self):
        xception = Xception(weights='imagenet', include_top=False,
                            input_shape=(None, None, 3))
        i = 0
        while(True):
            if xception.layers[i].name == 'add_2':
                break
            i += 1
        model = Model(inputs=xception.layers[0].input,
                      outputs=xception.layers[i].output,
                      name='xception')
        model.trainable = False
        model.compile('adadelta', 'mse')
        return model

    def get_dense_block(self, prev_filters, id):
        input_layer = Input(batch_shape=(None, None, None, prev_filters),
                            name='{}_{}_input'.format(self.dense_name, id))
        for i in range(1, 9):
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
        xception = self.get_xception()(preprocess)
        new_layers = self.get_dense_block(K.int_shape(xception)[-1], 0)(xception)
        return Model(inputs=input_layer,
                     outputs=new_layers,
                     name='{}'.format(self.low_name))

    def get_bottleneck(self, prev_filters, reduce):
        name = self.global_name if reduce else self.mid_name
        input_layer = Input(batch_shape=(None, None, None, prev_filters),
                            name='{}_input'.format(name))
        for i in range(1, 4):
            layer = input_layer if i == 1 else dense
            reduce_a = Conv2D(filters=K.int_shape(layer)[-1] // 2, kernel_size=1,
                              strides=1, padding='same',
                              kernel_initializer='he_uniform',
                              bias_initializer='he_uniform',
                              name='{}_reduce_{}a'.format(name, i))(layer)
            reduce_b = Conv2D(filters=K.int_shape(reduce_a)[-1], kernel_size=3,
                              strides=2 if reduce else 1, padding='same',
                              kernel_initializer='he_uniform',
                              bias_initializer='he_uniform',
                              name='{}_reduce_{}b'.format(name, i))(reduce_a)
            norm = BatchNormalization(scale=False,
                                      name='{}_reduce_{}norm'.format(name, i))(reduce_b)
            relu = Activation(activation='relu',
                              name='{}_reduce_{}relu'.format(name, i))(norm)
            dense = self.get_dense_block(K.int_shape(relu)[-1], i)(relu)
        output = GlobalMaxPooling2D(name='{}_pool'.format(name))(dense)
        return Model(inputs=input_layer,
                     outputs=output,
                     name='{}'.format(name))

    def __init__(self, name):
        self.global_name = '{}_global'.format(name)
        self.mid_name = '{}_mid'.format(name)
        self.low_name = '{}_low'.format(name)
        self.dense_name = '{}_dense'.format(name)
        self.feed_size = 256
