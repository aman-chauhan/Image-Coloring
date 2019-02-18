from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input

from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Dense

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

    def get_dense_block(self, prev_filters, id, count):
        input_layer = Input(batch_shape=(None, None, None, prev_filters),
                            name='{}_{}_input'.format(self.dense_name, id))
        for i in range(1, count + 1):
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
        new_layers = self.get_dense_block(K.int_shape(xception)[-1], 0, 8)(xception)
        return Model(inputs=input_layer,
                     outputs=new_layers,
                     name='{}'.format(self.low_name))

    def get_bottleneck(self, prev_filters, reduce):
        name = self.global_name if reduce else self.mid_name
        input_layer = Input(batch_shape=(None, None, None, prev_filters),
                            name='{}_input'.format(name))
        for i in range(1, 3):
            layer = input_layer if i == 1 else dense
            reduce_a = Conv2D(filters=K.int_shape(layer)[-1] // 2,
                              kernel_size=1, strides=1, padding='same',
                              kernel_initializer='he_uniform',
                              bias_initializer='he_uniform',
                              name='{}_reduce_{}a'.format(name, i))(layer)
            norm_a = BatchNormalization(scale=False,
                                        name='{}_reduce_{}a_norm'.format(name, i))(reduce_a)
            relu_a = Activation(activation='relu',
                                name='{}_reduce_{}a_relu'.format(name, i))(norm_a)
            reduce_b = Conv2D(filters=K.int_shape(relu_a)[-1], kernel_size=3,
                              strides=2 if reduce else 1, padding='same',
                              kernel_initializer='he_uniform',
                              bias_initializer='he_uniform',
                              name='{}_reduce_{}b'.format(name, i))(relu_a)
            norm_b = BatchNormalization(scale=False,
                                        name='{}_reduce_{}b_norm'.format(name, i))(reduce_b)
            relu_b = Activation(activation='relu',
                                name='{}_reduce_{}b_relu'.format(name, i))(norm_b)
            dense = self.get_dense_block(K.int_shape(relu_b)[-1], i, 8)(relu_b)
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
        reduce_a = Conv2D(filters=K.int_shape(fusion)[-1] // 2,
                          kernel_size=1, strides=1, padding='same',
                          kernel_initializer='he_uniform',
                          bias_initializer='he_uniform',
                          name='{}_reduce_a'.format(name))(fusion)
        norm_a = BatchNormalization(scale=False,
                                    name='{}_reduce_norm_a'.format(name))(reduce_a)
        relu_a = Activation(activation='relu',
                            name='{}_reduce_relu_a'.format(name))(norm_a)
        reduce_b = Conv2D(filters=K.int_shape(relu_a)[-1],
                          kernel_size=3, strides=1, padding='same',
                          kernel_initializer='he_uniform',
                          bias_initializer='he_uniform',
                          name='{}_reduce_b'.format(name))(relu_a)
        norm_b = BatchNormalization(scale=False,
                                    name='{}_reduce_norm_b'.format(name))(reduce_b)
        relu_b = Activation(activation='relu',
                            name='{}_reduce_relu_b'.format(name))(norm_b)
        return Model(inputs=[input_mid, input_global],
                     outputs=relu_b,
                     name='{}'.format(name))

    def get_classification(self, prev_filters, dropout=0.25):
        name = self.class_name
        input_layer = Input(batch_shape=(None, prev_filters),
                            name='{}_input'.format(name))
        full1 = Dense(units=2 * prev_filters,
                      kernel_initializer='he_uniform',
                      bias_initializer='he_uniform',
                      name='{}_full1'.format(name))(input_layer)
        full1_norm = BatchNormalization(scale=False,
                                        name='{}_full1_norm'.format(name))(full1)
        full1_relu = Activation(activation='relu',
                                name='{}_full1_relu'.format(name))(full1_norm)
        full1_drop = Dropout(rate=dropout,
                             name='{}_full1_drop'.format(name))(full1_relu)
        full2 = Dense(units=prev_filters,
                      kernel_initializer='he_uniform',
                      bias_initializer='he_uniform',
                      name='{}_full2'.format(name))(full1_drop)
        full2_norm = BatchNormalization(scale=False,
                                        name='{}_full2_norm'.format(name))(full2)
        full2_relu = Activation(activation='relu',
                                name='{}_full2_relu'.format(name))(full2_norm)
        full2_drop = Dropout(rate=dropout,
                             name='{}_full2_drop'.format(name))(full2_relu)
        output_layer = Dense(units=366, activation='softmax',
                             kernel_initializer='he_uniform',
                             bias_initializer='he_uniform',
                             name='{}_softmax'.format(name))(full2_drop)
        return Model(inputs=input_layer,
                     outputs=output_layer,
                     name='{}'.format(name))

    def get_color_features(self, prev_filters):
        name = self.color_name
        input_layer = Input(batch_shape=(None, None, None, prev_filters),
                            name='{}_input'.format(name))
        for i in range(1, 4):
            layer = input_layer if i == 1 else upsample
            reduce_a = Conv2D(filters=K.int_shape(layer)[-1] // 4,
                              kernel_size=3, strides=1, padding='same',
                              kernel_initializer='he_uniform',
                              bias_initializer='he_uniform',
                              name='{}_reduce_{}a'.format(name, i))(layer)
            norm_a = BatchNormalization(scale=False,
                                        name='{}_reduce_{}a_norm'.format(name, i))(reduce_a)
            relu_a = Activation(activation='relu',
                                name='{}_reduce_{}a_relu'.format(name, i))(norm_a)
            reduce_b = Conv2D(filters=K.int_shape(relu_a)[-1],
                              kernel_size=3, strides=1, padding='same',
                              kernel_initializer='he_uniform',
                              bias_initializer='he_uniform',
                              name='{}_reduce_{}b'.format(name, i))(relu_a)
            norm_b = BatchNormalization(scale=False,
                                        name='{}_reduce_{}b_norm'.format(name, i))(reduce_b)
            relu_b = Activation(activation='relu',
                                name='{}_reduce_{}b_relu'.format(name, i))(norm_b)
            upsample = UpSampling2D(size=(2, 2), name='{}_upsample_{}'.format(name, i))(relu_b)

        output_layer = Conv2D(filters=2, kernel_size=3, strides=1,
                              padding='same', activation='sigmoid',
                              kernel_initializer='he_uniform',
                              bias_initializer='he_uniform',
                              name='{}_sigmoid'.format(name))(upsample)
        return Model(inputs=input_layer,
                     outputs=output_layer,
                     name='{}'.format(name))

    def get_model(self):
        input_color = Input(batch_shape=(None, None, None, 1),
                            name='{}_color_input'.format(self.name))
        input_class = Input(batch_shape=(None, None, None, 1),
                            name='{}_class_input'.format(self.name))

        low_level_features = self.get_low_level_features()
        color_branch = low_level_features(input_color)
        class_branch = low_level_features(input_class)

        color_branch = self.get_bottleneck(K.int_shape(color_branch)[-1], False)(color_branch)
        class_branch = self.get_bottleneck(K.int_shape(class_branch)[-1], True)(class_branch)

        color_branch = self.get_fusion_features(K.int_shape(color_branch)[-1],
                                                K.int_shape(class_branch)[-1])([color_branch,
                                                                                class_branch])
        class_branch = self.get_classification(K.int_shape(class_branch)[-1])(class_branch)
        color_branch = self.get_color_features(K.int_shape(color_branch)[-1])(color_branch)
        return Model(inputs=[input_color, input_class],
                     outputs=[color_branch, class_branch],
                     name='{}'.format(self.name))

    def __init__(self, name):
        self.name = name
        self.color_name = '{}_color'.format(name)
        self.class_name = '{}_class'.format(name)
        self.fusion_name = '{}_fusion'.format(name)
        self.global_name = '{}_global'.format(name)
        self.mid_name = '{}_mid'.format(name)
        self.low_name = '{}_low'.format(name)
        self.dense_name = '{}_dense'.format(name)
        self.feed_size = 256
