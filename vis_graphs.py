from keras.utils import plot_model
from CNN import FullNetwork

plot_model(FullNetwork.model(), show_shapes=True, show_layer_names=True, to_file='docs/full_network.png')
plot_model(FullNetwork.llfn(), show_shapes=True, show_layer_names=True, to_file='docs/low_level_features_network.png')
plot_model(FullNetwork.mlfn(), show_shapes=True, show_layer_names=True, to_file='docs/mid_level_features_network.png')
plot_model(FullNetwork.gfn(), show_shapes=True, show_layer_names=True, to_file='docs/global_features_network.png')
plot_model(FullNetwork.clf(), show_shapes=True, show_layer_names=True, to_file='docs/classifier_network.png')
plot_model(FullNetwork.color(), show_shapes=True, show_layer_names=True, to_file='docs/color_network.png')
