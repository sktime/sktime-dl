__author__ = "James Large, Withington"

from tensorflow import keras

from sktime_dl.deeplearning.base.estimators import BaseDeepNetwork


class MCDCNNNetwork(BaseDeepNetwork):
    """Multi Channel Deep Convolutional Neural Network (MCDCNN).

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mcdcnn.py

    Network originally defined in:

    @inproceedings{zheng2014time, title={Time series classification using
    multi-channels deep convolutional neural networks}, author={Zheng,
    Yi and Liu, Qi and Chen, Enhong and Ge, Yong and Zhao, J Leon},
    booktitle={International Conference on Web-Age Information Management},
    pages={298--310}, year={2014}, organization={Springer} }
    """

    def __init__(
            self,
            kernel_size=5,
            pool_size=2,
            filter_sizes=[8, 8],
            dense_units=732,
            random_state=0,
    ):
        """
        :param kernel_size: int, specifying the length of the 1D convolution
         window
        :param pool_size: int, size of the max pooling windows
        :param filter_sizes: int, array of shape = 2, size of filter for each
         conv layer
        :param dense_units: int, number of units in the penultimate dense layer
        :param random_state: int, seed to any needed random actions
        """

        self.random_state = random_state

        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.filter_sizes = filter_sizes
        self.dense_units = dense_units

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
        n_t = input_shape[0]
        n_vars = input_shape[1]

        padding = "valid"

        if n_t < 60:  # for ItalyPowerOndemand
            padding = "same"

        input_layers = []
        conv2_layers = []

        for n_var in range(n_vars):
            input_layer = keras.layers.Input((n_t, 1))
            input_layers.append(input_layer)

            conv1_layer = keras.layers.Conv1D(
                self.filter_sizes[0],
                kernel_size=self.kernel_size,
                activation="relu",
                padding=padding,
            )(input_layer)
            conv1_layer = keras.layers.MaxPooling1D(pool_size=self.pool_size)(
                conv1_layer
            )

            conv2_layer = keras.layers.Conv1D(
                self.filter_sizes[1],
                kernel_size=self.kernel_size,
                activation="relu",
                padding=padding,
            )(conv1_layer)
            conv2_layer = keras.layers.MaxPooling1D(pool_size=self.pool_size)(
                conv2_layer
            )
            conv2_layer = keras.layers.Flatten()(conv2_layer)

            conv2_layers.append(conv2_layer)

        if n_vars == 1:
            # to work with univariate time series
            concat_layer = conv2_layers[0]
        else:
            concat_layer = keras.layers.Concatenate(axis=-1)(conv2_layers)

        fully_connected = keras.layers.Dense(
            units=self.dense_units, activation="relu"
        )(concat_layer)

        return input_layers, fully_connected

    def prepare_input(self, x):
        new_x = []
        n_vars = x.shape[2]

        for i in range(n_vars):
            new_x.append(x[:, :, i:i + 1])

        return new_x
