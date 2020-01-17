__author__ = "James Large, Withington"

import keras
import numpy as np
import pandas as pd

from keras import Sequential

from sktime_dl.networks.deeplearning._base import BaseDeepNetwork

class CNNNetwork(BaseDeepNetwork):
    """Time Convolutional Neural Network (CNN).

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/cnn.py

    Network originally defined in:

    @article{zhao2017convolutional,
      title={Convolutional neural networks for time series classification},
      author={Zhao, Bendong and Lu, Huanzhang and Chen, Shangfeng and Liu, Junliang and Wu, Dongya},
      journal={Journal of Systems Engineering and Electronics},
      volume={28},
      number={1},
      pages={162--169},
      year={2017},
      publisher={BIAI}
    }
    """

    def __init__(self,
                 nb_epochs=2000,
                 batch_size=16,
                 kernel_size=7,
                 avg_pool_size=3,
                 nb_conv_layers=2,
                 filter_sizes=[6, 12],
                 random_seed=0,
                 verbose=False,
                 model_name="cnn",
                 model_save_directory=None):
        '''
        :param nb_epochs: int, the number of epochs to train the model
        :param batch_size: int, the number of samples per gradient update.
        :param kernel_size: int, specifying the length of the 1D convolution window
        :param avg_pool_size: int, size of the average pooling windows
        :param nb_conv_layers: int, the number of convolutional plus average pooling layers
        :param filter_sizes: int, array of shape = (nb_conv_layers)
        :param random_seed: int, seed to any needed random actions
        :param verbose: boolean, whether to output extra information
        :param model_name: string, the name of this model for printing and file writing purposes
        :param model_save_directory: string, if not None; location to save the trained keras model in hdf5 format
        '''

        self.verbose = verbose
        self.model_name = model_name
        self.model_save_directory = model_save_directory
        self.is_fitted_ = False

        self.callbacks = []
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)

        self.input_shape = None
        self.model = None
        self.history = None

        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.nb_conv_layers = nb_conv_layers
        self.filter_sizes = filter_sizes

    def build_network(self, input_shape, **kwargs):
        """
        Construct un-compiled, un-trained, keras model layers
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        Returns
        -------
        output : input_layer, output_layer
        """
        padding = 'valid'
        input_layer = keras.layers.Input(input_shape)

        if input_shape[0] < 60:  # for italypowerondemand dataset
            padding = 'same'

        if len(self.filter_sizes) > self.nb_conv_layers:
            self.filter_sizes = self.filter_sizes[:self.nb_conv_layers]
        elif len(self.filter_sizes) < self.nb_conv_layers:
            self.filter_sizes = self.filter_sizes + [self.filter_sizes[-1]] * (
                    self.nb_conv_layers - len(self.filter_sizes))

        conv = keras.layers.Conv1D(filters=self.filter_sizes[0],
                                   kernel_size=self.kernel_size,
                                   padding=padding,
                                   activation='sigmoid')(input_layer)
        conv = keras.layers.AveragePooling1D(pool_size=self.avg_pool_size)(conv)

        for i in range(1, self.nb_conv_layers):
            conv = keras.layers.Conv1D(filters=self.filter_sizes[i],
                                       kernel_size=self.kernel_size,
                                       padding=padding,
                                       activation='sigmoid')(conv)
            conv = keras.layers.AveragePooling1D(pool_size=self.avg_pool_size)(conv)

        flatten_layer = keras.layers.Flatten()(conv)

        return input_layer, flatten_layer
