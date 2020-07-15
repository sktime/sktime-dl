__author__ = "Withington"

from tensorflow import keras

from sktime_dl.deeplearning.base.estimators import BaseDeepNetwork


class LSTMNetwork(BaseDeepNetwork):
    """ Long Short-Term Memory (LSTM)

    Adapted from the implementation of Brownlee, J. (2018)

    https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
    """

    def __init__(self):
        self.random_state = None
        self.units = None

    def build_network(self, input_shape, **kwargs):
        """
        Construct a network and return its input and output layers
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        input_layer = keras.layers.Input(input_shape)
        output_layer = keras.layers.LSTM(
            units=self.units[0],
            activation='relu',
            return_sequences=True)(input_layer)
        output_layer = keras.layers.LSTM(
            units=self.units[1],
            activation='relu')(output_layer)
        return input_layer, output_layer
