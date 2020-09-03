__author__ = "James Large, Withington"

from tensorflow import keras

from sktime_dl.deeplearning.base.estimators import BaseDeepNetwork


class MLPNetwork(BaseDeepNetwork):
    """Multi Layer Perceptron (MLP) (minus the final output layer).

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py

    Network originally defined in:

    @inproceedings{wang2017time, title={Time series classification from
    scratch with deep neural networks: A strong baseline}, author={Wang,
    Zhiguang and Yan, Weizhong and Oates, Tim}, booktitle={2017
    International joint conference on neural networks (IJCNN)}, pages={
    1578--1585}, year={2017}, organization={IEEE} }
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

        # flatten/reshape because when multivariate all should be on the
        # same axis
        input_layer_flattened = keras.layers.Flatten()(input_layer)

        layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
        layer_1 = keras.layers.Dense(500, activation="relu")(layer_1)

        layer_2 = keras.layers.Dropout(0.2)(layer_1)
        layer_2 = keras.layers.Dense(500, activation="relu")(layer_2)

        layer_3 = keras.layers.Dropout(0.2)(layer_2)
        layer_3 = keras.layers.Dense(500, activation="relu")(layer_3)

        output_layer = keras.layers.Dropout(0.3)(layer_3)

        return input_layer, output_layer
