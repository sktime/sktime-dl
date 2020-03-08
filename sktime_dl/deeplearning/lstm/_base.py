__author__ = "Withington"

from tensorflow import keras
import numpy as np

from sktime_dl.deeplearning.base.estimators import BaseDeepNetwork


class LSTMNetwork(BaseDeepNetwork):
    ''' Long Short-Term Memory (LSTM)

    Adapted from the implementation of Brownlee, J. (2018)
    
    https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
    '''
    def __init__(self,
                 units=[50, 50],
                 random_seed=0):
        '''
        :param units: int, array of size 2, the number of LSTM layers
        :param random_seed: int, seed to any needed random actions
        '''

        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)

        self.units = units

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
        output_layer = keras.layers.LSTM(units=self.units[0], activation='relu', return_sequences=True)(input_layer)
        output_layer = keras.layers.LSTM(units=self.units[1], activation='relu')(output_layer)
        return input_layer, output_layer
