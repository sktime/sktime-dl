__author__ = "James Large"

import numpy as np
from tensorflow import keras

from sktime_dl.deeplearning.base.estimators import BaseDeepNetwork


class MLSTMFCNNetwork(BaseDeepNetwork):
    """
    Multivariate Long Short Term Memory Fully Convolutional Network(MLSTMFCN)
    (minus the final output layer).

    Adapted from the implementation by Karim et. al

    https://github.com/houshd/MLSTM-FCN

    Network originally defined in:

    @article{Karim_2019,
    title={Multivariate LSTM-FCNs for time series classification},
    volume={116},
    ISSN={0893-6080},
    url={http://dx.doi.org/10.1016/j.neunet.2019.04.014},
    DOI={10.1016/j.neunet.2019.04.014},
    journal={Neural Networks},
    publisher={Elsevier BV},
    author={Karim, Fazle and Majumdar, Somshubra and
            Darabi, Houshang and Harford, Samuel},
    year={2019},
    month={Aug},
    pages={237–245}
    }
    """

    def __init__(self, nb_lstm_cells=64, random_seed=0):
        """
        :param random_seed: int, seed to any needed random actions
        """
        self.nb_lstm_cells = nb_lstm_cells

        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)

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

        # LSTM branch
        lstm_layers = keras.layers.Masking()(input_layer)
        lstm_layers = keras.layers.LSTM(self.nb_lstm_cells)(lstm_layers)
        lstm_layers = keras.layers.Dropout(0.8)(lstm_layers)

        # FCN branch
        fcn_layers = keras.layers.Permute((2, 1))(input_layer)
        fcn_layers = keras.layers.Conv1D(128, 8, padding='same',
                                         kernel_initializer='he_uniform')(
            fcn_layers)
        fcn_layers = keras.layers.BatchNormalization()(fcn_layers)
        fcn_layers = keras.layers.Activation('relu')(fcn_layers)
        fcn_layers = self.squeeze_excite_block(fcn_layers)

        fcn_layers = keras.layers.Conv1D(256, 5, padding='same',
                                         kernel_initializer='he_uniform')(
            fcn_layers)
        fcn_layers = keras.layers.BatchNormalization()(fcn_layers)
        fcn_layers = keras.layers.Activation('relu')(fcn_layers)
        fcn_layers = self.squeeze_excite_block(fcn_layers)

        fcn_layers = keras.layers.Conv1D(128, 3, padding='same',
                                         kernel_initializer='he_uniform')(
            fcn_layers)
        fcn_layers = keras.layers.BatchNormalization()(fcn_layers)
        fcn_layers = keras.layers.Activation('relu')(fcn_layers)

        fcn_layers = keras.layers.GlobalAveragePooling1D()(fcn_layers)

        # combination
        conc_layer = keras.layers.concatenate([lstm_layers, fcn_layers])

        return [input_layer, conc_layer]

    def squeeze_excite_block(self, input):
        '''
        Creates a squeeze-excite block and adds it onto the input layer
        ----------
        input : keras layer

        Returns
        ----------
        se_layer : keras layer
        '''

        # source code uses _keras_shape with keras
        # _shape_val is the tf.keras equivalent
        filters = input._shape_val[-1]

        se_layer = keras.layers.GlobalAveragePooling1D()(input)
        se_layer = keras.layers.Reshape((1, filters))(se_layer)
        se_layer = keras.layers.Dense(filters // 16, activation='relu',
                                      kernel_initializer='he_normal',
                                      use_bias=False)(se_layer)
        se_layer = keras.layers.Dense(filters, activation='sigmoid',
                                      kernel_initializer='he_normal',
                                      use_bias=False)(se_layer)
        se_layer = keras.layers.multiply([input, se_layer])
        return se_layer
