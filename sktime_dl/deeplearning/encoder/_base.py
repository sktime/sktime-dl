__author__ = "James Large, Withington"

import tensorflow
import tensorflow.keras as keras

from sktime_dl.deeplearning.base.estimators import BaseDeepNetwork

if tensorflow.__version__ >= "1.15" and tensorflow.__version__ <= "2":
    keras.__name__ = "tensorflow.keras"

if tensorflow.__version__ < "2.1.0":
    import keras_contrib as ADDONS
else:
    import tensorflow_addons as ADDONS


class EncoderNetwork(BaseDeepNetwork):
    """Encoder

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/encoder.py

    Network originally defined in:

    @article{serra2018towards,
       title={Towards a universal neural network encoder for time series},
       author={Serrà, J and Pascual, S and Karatzoglou, A},
       journal={Artif Intell Res Dev Curr Chall New Trends Appl},
       volume={308},
       pages={120},
       year={2018}
    }
    """

    def __init__(self, random_state=0):
        """
        :param random_state: int, seed to any needed random actions
        """
        self.random_state = random_state

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

        # conv block -1
        conv1 = keras.layers.Conv1D(
            filters=128, kernel_size=5, strides=1, padding="same"
        )(input_layer)
        conv1 = ADDONS.layers.InstanceNormalization()(conv1)
        conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = keras.layers.Dropout(rate=0.2)(conv1)
        conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
        # conv block -2
        conv2 = keras.layers.Conv1D(
            filters=256, kernel_size=11, strides=1, padding="same"
        )(conv1)
        conv2 = ADDONS.layers.InstanceNormalization()(conv2)
        conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = keras.layers.Dropout(rate=0.2)(conv2)
        conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        # conv block -3
        conv3 = keras.layers.Conv1D(
            filters=512, kernel_size=21, strides=1, padding="same"
        )(conv2)
        conv3 = ADDONS.layers.InstanceNormalization()(conv3)
        conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
        conv3 = keras.layers.Dropout(rate=0.2)(conv3)
        # split for attention
        attention_data = keras.layers.Lambda(lambda x: x[:, :, :256])(conv3)
        attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 256:])(conv3)
        # attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()(
            [attention_softmax, attention_data]
        )
        # last layer
        dense_layer = keras.layers.Dense(units=256, activation="sigmoid")(
            multiply_layer
        )
        dense_layer = ADDONS.layers.InstanceNormalization()(dense_layer)
        # output layer
        flatten_layer = keras.layers.Flatten()(dense_layer)

        return input_layer, flatten_layer
