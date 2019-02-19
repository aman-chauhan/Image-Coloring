from keras.utils import plot_model
import sys
import os


def print_shape(model_name):
    model = None
    if model_name == 'densenet':
        from models import IC_DenseNet121
        model = IC_DenseNet121('dn121').get_model()
    elif model_name == 'inceptionresnet':
        from models import IC_InceptionResNetV2
        model = IC_InceptionResNetV2('irv2').get_model()
    elif model_name == 'inception':
        from models import IC_InceptionV3
        model = IC_InceptionV3('iv3').get_model()
    elif model_name == 'resnet':
        from models import IC_ResNet50
        model = IC_ResNet50('r50').get_model()
    elif model_name == 'vgg':
        from models import IC_VGG19
        model = IC_VGG19('vgg').get_model()
    elif model_name == 'xception':
        from models import IC_Xception
        model = IC_Xception('x').get_model()

    if not os.path.exists(os.path.join('docs', model_name)):
        os.mkdir(os.path.join('docs', model_name))
    path = os.path.join('docs', model_name)
    plot_model(model, to_file=os.path.join(path, 'full.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
    plot_model(model.layers[2], to_file=os.path.join(path, 'low.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
    plot_model(model.layers[4], to_file=os.path.join(path, 'mid.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
    plot_model(model.layers[3], to_file=os.path.join(path, 'global.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
    plot_model(model.layers[-2], to_file=os.path.join(path, 'color.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
    plot_model(model.layers[-1], to_file=os.path.join(path, 'class.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
    plot_model(model.layers[2].layers[3],
               to_file=os.path.join(path, '{}.png'.format(model_name)),
               show_shapes=True, show_layer_names=True, rankdir='TB')
    plot_model(model.layers[2].layers[10],
               to_file=os.path.join(path, 'dense.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')


if __name__ == '__main__':
    print_shape(sys.argv[1])
