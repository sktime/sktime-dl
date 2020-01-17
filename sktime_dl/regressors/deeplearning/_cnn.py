__author__ = "James Large"

import keras
import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin

from sktime.regressors.base import BaseRegressor
from sktime.utils.validation.supervised import validate_X, validate_X_y



class CNNRegressor(BaseRegressor, RegressorMixin):
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

    # TODO reuse this function that is in classifiers _base.py
    def check_and_clean_data(self, X, y=None, input_checks=True):
        if input_checks:
            if y is None:
                validate_X(X)
            else:
                validate_X_y(X, y)

        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 1 and not isinstance(X.iloc[0, 0], pd.Series):
                raise TypeError(
                    "Input should either be a 2d numpy array, or a pandas dataframe with a single column of Series objects (networks cannot yet handle multivariate problems")
            elif X.shape[1] > 1:
                X = X.to_numpy()
            else:
                X = np.asarray([a.values for a in X.iloc[:, 0]])

        if len(X.shape) == 2:
            # add a dimension to make it multivariate with one dimension
            X = X.reshape((X.shape[0], X.shape[1], 1))

        return X

    def build_model(self, input_shape, nb_classes, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for training
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        nb_classes: int
            The number of classes, which shall become the size of the output layer
        Returns
        -------
        output : a compiled Keras Model
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
        #output_layer = keras.layers.Dense(units=nb_classes, activation='sigmoid')(flatten_layer)
        output_layer = keras.layers.Dense(units=1)(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        return model

    def fit(self, X, y, input_checks=True, **kwargs):
        """
        Build the classifier on the training set (X, y)
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame is passed, column 0 is extracted.
        y : array-like, shape = [n_instances]
            The class labels.
        input_checks: boolean
            whether to check the X and y parameters
        Returns
        -------
        self : object
        """
        X = self.check_and_clean_data(X, y, input_checks=input_checks)

        #y_onehot = self.convert_y(y)
        self.input_shape = X.shape[1:]

        #self.model = self.build_model(self.input_shape, self.nb_classes)
        self.model = self.build_model(self.input_shape, nb_classes=1) # Remove need for nb_classes

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(X, y, batch_size=self.batch_size, epochs=self.nb_epochs,
                                      verbose=self.verbose, callbacks=self.callbacks)

        #self.save_trained_model()
        self.is_fitted_ = True

        return self

    def predict(self, X, input_checks=True, **kwargs):
        """
        Find regression estimate for all cases in X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.
            If a Pandas data frame is passed (sktime format)
            If a Pandas data frame is passed, a check is performed that it only has one column.
            If not, an exception is thrown, since this classifier does not yet have
            multivariate capability.
        input_checks: boolean
            whether to check the X parameter
        Returns
        -------
        predictions : 1d numpy array
            array of predictions of each instance
        """
        # TODO add check_is_fitted

        print('_cnn_regressor X.shape before:', X.shape) # TODO remove
        X = self.check_and_clean_data(X, input_checks=input_checks)
        print('_cnn_regressor X.shape after:', X.shape) # TODO remove

        y_pred = self.model.predict(X, **kwargs)
        if y_pred.ndim == 1:
            y_pred.ravel()
        return y_pred
