# Base class for regressors

__author__ = "Withington"

import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted

from sktime.regressors.base import BaseRegressor
from sktime.utils.validation.supervised import validate_X, validate_X_y

from sktime_dl.utils import save_trained_model


class BaseDeepRegressor(BaseRegressor, RegressorMixin):

    def __init__(self,
                 model_name=None,
                 model_save_directory=None):
        self.model_save_directory = model_save_directory
        self.model = None
        self.model_name = model_name

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
        raise NotImplementedError('this is an abstract method')

    def predict(self, X, input_checks=True, **kwargs):
        """
        Find regression estimate for all cases in X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.
            If a Pandas data frame of Series objects is passed (sktime format), 
            a check is performed that it only has one column.
            If not, an exception is thrown, since this regressor does not yet have
            multivariate capability.
        input_checks: boolean
            whether to check the X parameter
        Returns
        -------
        predictions : 1d numpy array
            array of predictions of each instance
        """
        check_is_fitted(self)

        X = self.check_and_clean_data(X, input_checks=input_checks)

        y_pred = self.model.predict(X, **kwargs)

        if y_pred.ndim == 1:
            y_pred.ravel()
        return y_pred

    def check_and_clean_data(self, X, y=None, input_checks=True):
        if input_checks:
            if y is None:
                validate_X(X)
            else:
                validate_X_y(X, y)

        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 1 and not isinstance(X.iloc[0, 0], pd.Series):
                raise TypeError(
                    "Input should either be a 2D array of values or a pandas dataframe with a single column of Series objects (networks cannot yet handle multivariate problems")
            elif X.shape[1] > 1:
                X = X.to_numpy()
            else:
                X = np.asarray([a.values for a in X.iloc[:, 0]])

        if len(X.shape) == 2:
            # add a dimension to make it multivariate with one dimension
            X = X.reshape((X.shape[0], X.shape[1], 1))

        return X

    def save_trained_model(self):
        save_trained_model(
            self.model,
            self.model_save_directory,
            self.model_name)
