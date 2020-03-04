#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["SimpleRNNRegressor"]

import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from sklearn.utils.validation import check_is_fitted
from sktime_dl.deeplearning.base.estimators import BaseDeepRegressor, BaseDeepNetwork
from sktime_dl.utils import check_and_clean_data


class SimpleRNNRegressor(BaseDeepRegressor, BaseDeepNetwork):

    def __init__(self, nb_epochs=100, batch_size=1, callback=None, random_seed=0, verbose=0,
                 model_name="simple_rnn_regressor", model_save_directory=None):
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.is_fitted_ = False
        self.callbacks = callback if callback is not None else []

        self.model = None
        self.history = None
        self.input_shape = None

        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)

        super(SimpleRNNRegressor, self).__init__(
            model_name=model_name,
            model_save_directory=model_save_directory)

    def build_model(self, input_shape, **kwargs):
        model = Sequential([
            SimpleRNN(6, input_shape=(input_shape, 1), activation='linear',
                      use_bias=False, kernel_initializer='glorot_uniform',
                      recurrent_initializer='orthogonal', bias_initializer='zeros',
                      dropout=0.0, recurrent_dropout=0.0),
            Dense(1, use_bias=True, activation='linear')
        ])
        model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.001))
        if not any(isinstance(callback, ReduceLROnPlateau) for callback in self.callbacks):
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                          min_lr=0.0001)
            self.callbacks.append(reduce_lr)
        return model

    def fit(self, X, y, input_checks=True, **kwargs):
        X = check_and_clean_data(X, y, input_checks=input_checks)
        self._input_shape = X.shape[1:]
        self.batch_size = int(max(1, min(X.shape[0] / 10, self.batch_size)))

        self.model = self.build_model(self.input_shape)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(X, y, batch_size=self.batch_size, epochs=self.nb_epochs,
                                      verbose=self.verbose, callbacks=self.callbacks)
        self.save_trained_model()
        self.is_fitted_ = True
        return self

    def predict(self, X, input_checks=True, **kwargs):
        check_is_fitted(self)

        X = check_and_clean_data(X, input_checks=input_checks)

        y_pred = self.model.predict(X, **kwargs)

        if y_pred.ndim == 1:
            y_pred.ravel()
        return y_pred
