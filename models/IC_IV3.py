from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

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


def get_inception():
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


def get_dense_block(name, prev_filters, id, count):
    name = '{}_dense'.format(name)
    input_layer = Input(batch_shape=(None, None, None, prev_filters),
                        name='{}_{}_input'.format(name, id))
    for i in range(1, count + 1):
        conv = Conv2D(filters=64, kernel_size=1, strides=1, padding='same',
                      kernel_initializer='he_uniform',
                      bias_initializer='he_uniform',
                      name='{}_{}_conv_{}a'.format(name, id, i))
        conv1 = conv(input_layer if i == 1 else layer)
        norm1 = BatchNormalization(scale=False,
                                   name='{}_{}_norm_{}a'.format(name, id, i))(conv1)
        relu1 = Activation(activation='relu',
                           name='{}_{}_relu_{}a'.format(name, id, i))(norm1)
        conv2 = Conv2D(filters=16, kernel_size=3, strides=1, padding='same',
                       kernel_initializer='he_uniform',
                       bias_initializer='he_uniform',
                       name='{}_{}_conv_{}b'.format(name, id, i))(relu1)
        norm2 = BatchNormalization(scale=False,
                                   name='{}_{}_norm_{}b'.format(name, id, i))(conv2)
        relu2 = Activation(activation='relu',
                           name='{}_{}_relu_{}b'.format(name, id, i))(norm2)
        concat = Concatenate(name='{}_{}_concat_{}'.format(name, id, i))
        layer = concat([input_layer if i == 1 else layer, relu2])
    return Model(inputs=input_layer,
                 outputs=layer,
                 name='{}_{}'.format(name, id))


def copy_dims(x):
    return K.tile(x, [1, 1, 1, 3])


def get_low_level_features(name):
    name = '{}_low'.format(name)
    input_layer = Input(batch_shape=(None, None, None, 1),
                        name='{}_input'.format(name))
    duplicate = Lambda(copy_dims, name='{}_copy'.format(name))(input_layer)
    preprocess = Lambda(preprocess_input, name='{}_pre'.format(name))(duplicate)
    inception = get_inception()(preprocess)
    reduce_a = Conv2D(filters=128, kernel_size=1, strides=1, padding='same',
                      kernel_initializer='he_uniform',
                      bias_initializer='he_uniform',
                      name='{}_reduce_a'.format(name))(inception)
    norm_a = BatchNormalization(scale=False,
                                name='{}_reduce_a_norm'.format(name))(reduce_a)
    relu_a = Activation(activation='relu',
                        name='{}_reduce_a_relu'.format(name))(norm_a)
    reduce_b = Conv2D(filters=128, kernel_size=3, strides=1, padding='same',
                      kernel_initializer='he_uniform',
                      bias_initializer='he_uniform',
                      name='{}_reduce_b'.format(name))(relu_a)
    norm_b = BatchNormalization(scale=False,
                                name='{}_reduce_b_norm'.format(name))(reduce_b)
    relu_b = Activation(activation='relu',
                        name='{}_reduce_b_relu'.format(name))(norm_b)
    new_layers = get_dense_block(name, K.int_shape(relu_b)[-1], 0, 8)(relu_b)
    return Model(inputs=input_layer,
                 outputs=new_layers,
                 name='{}'.format(name))


def get_bottleneck(name, prev_filters, reduce):
    name = '{}_global'.format(name) if reduce else '{}_mid'.format(name)
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
        dense = get_dense_block(name, K.int_shape(relu_b)[-1], i, 8)(relu_b)
    if reduce:
        output = GlobalMaxPooling2D(name='{}_pool'.format(name))(dense)
    else:
        output = dense
    return Model(inputs=input_layer,
                 outputs=output,
                 name='{}'.format(name))


def get_classification(name, prev_filters, dropout=0.25):
    name = '{}_class'.format(name)
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
    output_layer = Dense(units=365, activation='softmax',
                         kernel_initializer='he_uniform',
                         bias_initializer='he_uniform',
                         name='{}_softmax'.format(name))(full2_drop)
    return Model(inputs=input_layer,
                 outputs=output_layer,
                 name='{}'.format(name))


def get_color_features(name, prev_filters):
    name = '{}_color'.format(name)
    input_layer = Input(batch_shape=(None, None, None, prev_filters),
                        name='{}_input'.format(name))
    for i in range(1, 4):
        layer = input_layer if i == 1 else upsample
        reduce_a = Conv2D(filters=K.int_shape(layer)[-1] // 2,
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


def tile_vector(x, k):
    return K.tile(K.expand_dims(K.expand_dims(x, 1), 1), [1, k[1], k[2], 1])


def get_model(name):
    input_color = Input(batch_shape=(None, None, None, 1),
                        name='{}_color_input'.format(name))
    input_class = Input(batch_shape=(None, None, None, 1),
                        name='{}_class_input'.format(name))

    low_features = get_low_level_features(name)
    color_branch = low_features(input_color)
    class_branch = low_features(input_class)

    color_branch = get_bottleneck(name, K.int_shape(color_branch)[-1], False)(color_branch)
    class_branch = get_bottleneck(name, K.int_shape(class_branch)[-1], True)(class_branch)

    tile = Lambda(tile_vector, arguments={'k': K.shape(color_branch)},
                  name='{}_global_tile'.format(name))(class_branch)
    fusion = Concatenate(name='{}_concat'.format(name))([color_branch, tile])
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
    color_branch = Activation(activation='relu',
                              name='{}_reduce_relu_b'.format(name))(norm_b)

    color_branch = get_color_features(name, K.int_shape(color_branch)[-1])(color_branch)
    class_branch = get_classification(name, K.int_shape(class_branch)[-1])(class_branch)
    return Model(inputs=[input_color, input_class],
                 outputs=[color_branch, class_branch],
                 name='{}'.format(name))
