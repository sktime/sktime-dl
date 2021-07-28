__author__ = "Jack Russon"
import tensorflow as tf
from tensorflow import keras

from sktime_dl.networks._network import BaseDeepNetwork


class MACNNNetwork(BaseDeepNetwork):
    """
    """

    def __init__(
            self,
            random_state=1,
            padding='same'
    ):
        """
        :param kernel_sizes: list of ints, specifying the length of the 1D convolution
         windows
        :param filter_sizes: int, array of shape = 3, size of filter for each
         conv layer
        :param random_state: int, seed to any needed random actions
        """
        super(MACNNNetwork, self).__init__()
        self.random_state=random_state
        self.padding=padding



    def __MACNN_block(self, x, kernels, reduce):
        cov1 = keras.layers.Conv1D(kernels, 3, padding='same')(x)

        cov2 = keras.layers.Conv1D(kernels, 6, padding='same')(x)

        cov3 = keras.layers.Conv1D(kernels, 12, padding='same')(x)

        x = keras.layers.Concatenate(axis=2)([cov1, cov2, cov3])
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        y = tf.math.reduce_mean(x, 1)
        y = keras.layers.Dense(int(kernels * 3 / reduce), use_bias=False, activation='relu')(y)
        y = keras.layers.Dense(int(kernels * 3), use_bias=False, activation='relu')(y)
        y = tf.reshape(y, [-1, 1, kernels * 3])
        return x * y

    def __stack(self, x, loop_num, kernels, reduce=16):
        for i in range(loop_num):
            x = self.__MACNN_block(x, kernels, reduce)
        return x


    def build_network(self, input_shape, **kwargs):
        """
        Construct a network and return its input and output layers
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        Returns
        -------
        input_layers : keras layers
        output_layer : a keras layer
        """
        input_layer = keras.layers.Input(shape=input_shape)
        x=self.__stack(input_layer,2,64,16)
        x = keras.layers.MaxPooling1D( 3, 2, padding='same')(x)
        x = self.__stack(x, 2, 128, 16)
        x = keras.layers.MaxPooling1D( 3, 2, padding='same')(x)
        x = self.__stack(x, 2, 256, 16)

        fc = tf.reduce_mean(x, 1)

        return input_layer, fc


