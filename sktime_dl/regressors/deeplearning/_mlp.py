__author__ = "James Large, Withington"

import keras
import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin

from sktime.utils.validation.supervised import validate_X, validate_X_y

from sktime_dl.regressors.deeplearning._base import BaseDeepRegressor
from sktime_dl.networks.deeplearning import MLPNetwork


class MLPRegressor(BaseDeepRegressor, RegressorMixin):
    """Multi Layer Perceptron (MLP).

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py

    Network originally defined in:

    @inproceedings{wang2017time,
      title={Time series classification from scratch with deep neural networks: A strong baseline},
      author={Wang, Zhiguang and Yan, Weizhong and Oates, Tim},
      booktitle={2017 International joint conference on neural networks (IJCNN)},
      pages={1578--1585},
      year={2017},
      organization={IEEE}
    }
    """

    def __init__(self,
                 nb_epochs=2000,
                 batch_size=16,

                 random_seed=0,
                 verbose=False,
                 model_name="mlp_regressor",
                 model_save_directory=None):
        '''
        :param nb_epochs: int, the number of epochs to train the model
        :param batch_size: int, the number of samples per gradient update.
        :param random_seed: int, seed to any needed random actions
        :param verbose: boolean, whether to output extra information
        :param model_name: string, the name of this model for printing and file writing purposes
        :param model_save_directory: string, if not None; location to save the trained keras model in hdf5 format
        '''

        self.verbose = verbose
        self.model_name = model_name
        self.model_save_directory = model_save_directory
        self.is_fitted_ = False

        # calced in fit
        self.classes_ = None
        self.nb_classes = -1
        self.input_shape = None
        self.model = None
        self.history = None

        # predefined
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.callbacks = None

        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)

    def build_model(self, input_shape, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for training
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        Returns
        -------
        output : a compiled Keras Model
        """
        network = MLPNetwork(self.random_seed)
        input_layer, output_layer = network.build_network(input_shape, **kwargs)
        output_layer = keras.layers.Dense(units=1)(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

        self.callbacks = [reduce_lr]

        return model

    def fit(self, X, y, input_checks=True, **kwargs):
        """
        Build the regressor on the training set (X, y)
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame is passed, column 0 is extracted.
        y : array-like, shape = [n_instances]
            The regression values.
        input_checks: boolean
            whether to check the X and y parameters
        Returns
        -------
        self : object
        """
        X = self.check_and_clean_data(X, y, input_checks=input_checks)
        self.input_shape = X.shape[1:]

        self.batch_size = int(max(1, min(X.shape[0] / 10, self.batch_size)))

        self.model = self.build_model(self.input_shape)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(X, y, batch_size=self.batch_size, epochs=self.nb_epochs,
                                      verbose=self.verbose, callbacks=self.callbacks)

        self.save_trained_model()
        self.is_fitted_ = True

        return self