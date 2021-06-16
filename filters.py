
import tensorflow as tf
import pathlib as pl
import matplotlib.pyplot as plt

from model import create_model


def visualize_filters():
    """
    Visualize convolutional filters by maximizing response with
    gradient descent, starting with a blank image
    """

    path = pl.Path("checkpoints")
    cp = path / pl.Path("128.hdf5")

    image_size = (180, 180)
    n_classes = 5

    model = create_model(image_size, n_classes, weights=cp)

    layer = model.layers[0]

    channels = layer.output_shape[-1]

    images = 0.5 + tf.zeros((channels, *image_size, 3))

    activations = model.predict(images)

    # gradient descent?
    # https://towardsdatascience.com/understanding-your-convolution-network-with-visualizations-a4883441533b
    # https://github.com/anktplwl91/visualizing_convnets/blob/master/model_training_and_visualizations.py
    # https://distill.pub/2017/feature-visualization/

    print(activations.shape)
