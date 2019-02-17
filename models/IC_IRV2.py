from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input

from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import Input

from keras.models import Model
from keras import backend as K


class IC_InceptionResNetV2:
    def get_resnet(self):
        resnet = InceptionResNetV2(weights='imagenet', include_top=False,
                                   input_shape=(None, None, 3))
        i = 0
        while(True):
            if resnet.layers[i].name == 'block35_3_ac':
                break
            i += 1
        model = Model(inputs=resnet.layers[0].input,
                      outputs=resnet.layers[i].output,
                      name='incept_resnet')
        model.trainable = False
        model.compile('adadelta', 'mse')
        return model

    def get_dense_block(self, prev_filters, id):
        input_layer = Input(batch_shape=(None, None, None, prev_filters),
                            name='{}_{}_input'.format(self.dense_name, id))
        for i in range(1, 7):
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
        resnet = self.get_resnet()(preprocess)
        new_layers = self.get_dense_block(K.int_shape(resnet)[-1], 0)(resnet)
        return Model(inputs=input_layer,
                     outputs=new_layers,
                     name='{}'.format(self.low_name))

    def get_bottleneck(self, prev_filters, reduce):
        name = self.global_name if reduce else self.mid_name
        input_layer = Input(batch_shape=(None, None, None, prev_filters),
                            name='{}_input'.format(name))
        for i in range(1, 3):
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
        if reduce:
            output = GlobalMaxPooling2D(name='{}_pool'.format(name))(dense)
        else:
            output = dense
        return Model(inputs=input_layer,
                     outputs=output,
                     name='{}'.format(name))

    def get_fusion_features(self, prev_mid, prev_global):
        name = self.fusion_name
        input_mid = Input(batch_shape=(None, None, None, prev_mid),
                          name='{}_mid_input'.format(name))
        input_global = Input(batch_shape=(None, prev_global),
                             name='{}_global_input'.format(name))
        tile = Lambda(lambda x, k: K.tile(K.expand_dims(K.expand_dims(x, 1), 1),
                                          [1, k[1], k[2], 1]),
                      arguments={'k': K.shape(input_mid)},
                      name='{}_global_tile'.format(name))(input_global)
        fusion = Concatenate(name='{}_concat'.format(name))([input_mid, tile])
        reduce_a = Conv2D(filters=K.int_shape(fusion)[-1] // 2, kernel_size=1,
                          strides=1, padding='same',
                          kernel_initializer='he_uniform',
                          bias_initializer='he_uniform',
                          name='{}_reduce_a'.format(name))(fusion)
        reduce_b = Conv2D(filters=K.int_shape(reduce_a)[-1], kernel_size=3,
                          strides=1, padding='same',
                          kernel_initializer='he_uniform',
                          bias_initializer='he_uniform',
                          name='{}_reduce_b'.format(name))(reduce_a)
        norm = BatchNormalization(scale=False,
                                  name='{}_reduce_norm'.format(name))(reduce_b)
        relu = Activation(activation='relu',
                          name='{}_reduce_relu'.format(name))(norm)
        return Model(inputs=[input_mid, input_global],
                     outputs=relu,
                     name='{}'.format(name))

    def get_model(self):
        input_color = Input(batch_shape=(None, None, None, 1),
                            name='{}_color'.format(self.name))
        input_class = Input(batch_shape=(None, None, None, 1),
                            name='{}_class'.format(self.name))

        low_level_features = self.get_low_level_features()
        color_branch = low_level_features(input_color)
        class_branch = low_level_features(input_class)

        color_branch = self.get_bottleneck(K.int_shape(color_branch)[-1], False)(color_branch)
        class_branch = self.get_bottleneck(K.int_shape(class_branch)[-1], True)(class_branch)

        output = self.get_fusion_features(K.int_shape(color_branch)[-1],
                                          K.int_shape(class_branch)[-1])([color_branch,
                                                                          class_branch])
        return Model(inputs=[input_color, input_class],
                     outputs=output,
                     name='{}'.format(self.name))

    def __init__(self, name):
        self.name = name
        self.fusion_name = '{}_fusion'.format(name)
        self.global_name = '{}_global'.format(name)
        self.mid_name = '{}_mid'.format(name)
        self.low_name = '{}_low'.format(name)
        self.dense_name = '{}_dense'.format(name)
        self.feed_size = 278
