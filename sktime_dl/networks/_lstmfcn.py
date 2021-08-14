__author__ = "Jack Russon"

from tensorflow import keras
from sktime_dl.utils.layer_utils import AttentionLSTM

from sktime_dl.networks._network import BaseDeepNetwork


class LSTMFCNNetwork(BaseDeepNetwork):
    """
    """

    def __init__(
            self,
            kernel_sizes=[8, 5, 3],
            filter_sizes=[128, 256, 128],
            NUM_CELLS=8,
            random_state=0,
            dropout=0.8,
            attention=False
    ):
        """
        :param kernel_sizes: list of ints, specifying the length of the 1D convolution
         windows
        :param filter_sizes: int, array of shape = 3, size of filter for each
         conv layer
        :param random_state: int, seed to any needed random actions
        """

        self.random_state = random_state
        self.kernel_sizes = kernel_sizes
        self.filter_sizes = filter_sizes
        self.NUM_CELLS=NUM_CELLS
        self.dropout=dropout
        self.attention=attention

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

        x = keras.layers.Permute((2, 1))(input_layer)
        if self.attention:

            x = AttentionLSTM(self.NUM_CELLS)(x)
        else:
            x = keras.layers.LSTM(self.NUM_CELLS)(x)
        x = keras.layers.Dropout(self.dropout)(x)

        y = keras.layers.Conv1D(self.filter_sizes[0], self.kernel_sizes[0], padding='same', kernel_initializer='he_uniform')(input_layer)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation('relu')(y)

        y = keras.layers.Conv1D(self.filter_sizes[1], self.kernel_sizes[1], padding='same', kernel_initializer='he_uniform')(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation('relu')(y)

        y = keras.layers.Conv1D(self.filter_sizes[2], self.kernel_sizes[2], padding='same', kernel_initializer='he_uniform')(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation('relu')(y)

        y = keras.layers.GlobalAveragePooling1D()(y)

        output_layer = keras.layers.concatenate([x, y])

        return input_layer, output_layer

