__author__ = "James Large, Withington"

from tensorflow import keras

from sktime_dl.deeplearning.base.estimators import BaseDeepNetwork


class FCNNetwork(BaseDeepNetwork):
    """Fully convolutional neural network (FCN).

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py

    Network originally defined in:

    @inproceedings{wang2017time,
      title={Time series classification from scratch with deep neural networks:
       A strong baseline},
      author={Wang, Zhiguang and Yan, Weizhong and Oates, Tim},
      booktitle={2017 International joint conference on neural networks
      (IJCNN)},
      pages={1578--1585},
      year={2017},
      organization={IEEE}
    }

    :param random_state: int, seed to any needed random actions
    """

    def __init__(self, random_state=0):
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

        conv1 = keras.layers.Conv1D(
            filters=128, kernel_size=8, padding="same"
        )(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation="relu")(conv1)

        conv2 = keras.layers.Conv1D(
            filters=256, kernel_size=5, padding="same"
        )(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation("relu")(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3, padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation("relu")(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        return input_layer, gap_layer
