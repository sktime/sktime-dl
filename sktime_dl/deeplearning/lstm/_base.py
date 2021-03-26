__author__ = "Mahak Kothari"

from tensorflow import keras

from sktime_dl.deeplearning.base.estimators import BaseDeepNetwork


class LSTMNetwork(BaseDeepNetwork):
    """ 
    Long Short-Term Memory (LSTM)

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
                dropout=0.2, recurrent_dropout=0.2,              
                return_sequences=True)(input_layer)

        for i in range(len(units)-2):            
            output_layer = keras.layers.LSTM(
                units=self.units[i+1],
                dropout=0.2, recurrent_dropout=0.2,
                return_sequences = True)(output_layer)

        output_layer = keras.layers.LSTM(
                units=self.units[len(units)-1],
                dropout=0.2, recurrent_dropout=0.2,
                return_sequences = False)(output_layer)
        return input_layer, output_layer
















