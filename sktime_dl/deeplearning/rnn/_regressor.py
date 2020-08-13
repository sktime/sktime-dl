#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["SimpleRNNRegressor"]

from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

from sktime_dl.deeplearning.base.estimators import BaseDeepNetwork
from sktime_dl.deeplearning.base.estimators import BaseDeepRegressor
from sktime_dl.utils import check_and_clean_data, \
    check_and_clean_validation_data


class SimpleRNNRegressor(BaseDeepRegressor, BaseDeepNetwork):
    """Simple recurrent neural network

    References
    ----------
    ..[1] benchmark forecaster in M4 forecasting competition:
    https://github.com/Mcompetitions/M4-methods
    """

    def __init__(
            self,
            nb_epochs=100,
            batch_size=1,
            units=6,
            callbacks=None,
            random_state=0,
            verbose=0,
            model_name="simple_rnn_regressor",
            model_save_directory=None,
    ):
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.units = units
        self.callbacks = callbacks
        self.random_state = random_state
        super(SimpleRNNRegressor, self).__init__(
            model_name=model_name,
            model_save_directory=model_save_directory
        )

    def build_model(self, input_shape, **kwargs):
        model = Sequential(
            [
                SimpleRNN(
                    self.units,
                    input_shape=input_shape,
                    activation="linear",
                    use_bias=False,
                    kernel_initializer="glorot_uniform",
                    recurrent_initializer="orthogonal",
                    bias_initializer="zeros",
                    dropout=0.0,
                    recurrent_dropout=0.0,
                ),
                Dense(1, use_bias=True, activation="linear"),
            ]
        )
        model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.001))

        if self.callbacks is None:
            self.callbacks = []

        if not any(
                isinstance(callback, ReduceLROnPlateau)
                for callback in self.callbacks
        ):
            reduce_lr = ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=50, min_lr=0.0001
            )
            self.callbacks.append(reduce_lr)
        return model

    def fit(self, X, y, input_checks=True, validation_X=None,
            validation_y=None, **kwargs):
        """
        Fit the regressor on the training set (X, y)
        ----------
        X : a nested pd.Dataframe, or (if input_checks=False) array-like of
        shape = (n_instances, series_length, n_dimensions)
            The training input samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
        y : array-like, shape = [n_instances]
            The training data class labels.
        input_checks : boolean
            whether to check the X and y parameters
        validation_X : a nested pd.Dataframe, or array-like of shape =
        (n_instances, series_length, n_dimensions)
            The validation samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
            Unless strictly defined by the user via callbacks (such as
            EarlyStopping), the presence or state of the validation
            data does not alter training in any way. Predictions at each epoch
            are stored in the model's fit history.
        validation_y : array-like, shape = [n_instances]
            The validation class labels.
        Returns
        -------
        self : object
        """
        X = check_and_clean_data(X, y, input_checks=input_checks)

        validation_data = \
            check_and_clean_validation_data(validation_X, validation_y)

        self.input_shape = X.shape[1:]
        self.batch_size = int(max(1, min(X.shape[0] / 10, self.batch_size)))

        self.model = self.build_model(self.input_shape)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
            validation_data=validation_data,
        )
        self.save_trained_model()
        self._is_fitted = True
        return self
