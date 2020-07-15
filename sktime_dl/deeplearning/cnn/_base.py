__author__ = "James Large, Withington"

from tensorflow import keras

from sktime_dl.deeplearning.base.estimators import BaseDeepNetwork


class CNNNetwork(BaseDeepNetwork):
    """Time Convolutional Neural Network (CNN) (minus the final output layer).

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/cnn.py

    Network originally defined in:

    @article{zhao2017convolutional,
      title={Convolutional neural networks for time series classification},
      author={Zhao, Bendong and Lu, Huanzhang and Chen, Shangfeng and Liu,
      Junliang and Wu, Dongya},
      journal={Journal of Systems Engineering and Electronics},
      volume={28},
      number={1},
      pages={162--169},
      year={2017},
      publisher={BIAI}
    }
    """

    def __init__(
            self,
            kernel_size=7,
            avg_pool_size=3,
            nb_conv_layers=2,
            filter_sizes=[6, 12],
            random_state=0,
    ):
        """
        :param kernel_size: int, specifying the length of the 1D convolution
         window
        :param avg_pool_size: int, size of the average pooling windows
        :param nb_conv_layers: int, the number of convolutional plus average
         pooling layers
        :param filter_sizes: int, array of shape = (nb_conv_layers)
        :param random_state: int, seed to any needed random actions
        """

        self.random_state = random_state
        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.nb_conv_layers = nb_conv_layers
        self.filter_sizes = filter_sizes

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
        padding = "valid"
        input_layer = keras.layers.Input(input_shape)

        if input_shape[0] < 60:  # for ItalyPowerDemand dataset
            padding = "same"

        if len(self.filter_sizes) > self.nb_conv_layers:
            self.filter_sizes = self.filter_sizes[: self.nb_conv_layers]
        elif len(self.filter_sizes) < self.nb_conv_layers:
            self.filter_sizes = self.filter_sizes + [self.filter_sizes[-1]] * (
                    self.nb_conv_layers - len(self.filter_sizes)
            )

        conv = keras.layers.Conv1D(
            filters=self.filter_sizes[0],
            kernel_size=self.kernel_size,
            padding=padding,
            activation="sigmoid",
        )(input_layer)
        conv = keras.layers.AveragePooling1D(pool_size=self.avg_pool_size)(
            conv
        )

        for i in range(1, self.nb_conv_layers):
            conv = keras.layers.Conv1D(
                filters=self.filter_sizes[i],
                kernel_size=self.kernel_size,
                padding=padding,
                activation="sigmoid",
            )(conv)
            conv = keras.layers.AveragePooling1D(pool_size=self.avg_pool_size)(
                conv
            )

        flatten_layer = keras.layers.Flatten()(conv)

        return input_layer, flatten_layer
