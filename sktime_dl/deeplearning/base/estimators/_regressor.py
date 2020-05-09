# Base class for regressors

__author__ = "Withington, James Large"

from sklearn.base import RegressorMixin
from sktime.regressors.base import BaseRegressor

from sktime_dl.utils import check_and_clean_data
from sktime_dl.utils import check_is_fitted
from sktime_dl.utils import save_trained_model


class BaseDeepRegressor(BaseRegressor, RegressorMixin):
    def __init__(self, model_name=None, model_save_directory=None):
        self.model_save_directory = model_save_directory
        self.model = None
        self.model_name = model_name

    def build_model(self, input_shape, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for
        training ---------- input_shape : tuple The shape of the data fed
        into the input layer Returns ------- output : a compiled Keras Model
        """
        raise NotImplementedError("this is an abstract method")

    def predict(self, X, input_checks=True, **kwargs):
        """
        Find regression estimate for all cases in X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.
            If a Pandas data frame of Series objects is passed (sktime
            format), a check is performed that it only has one column.
            If not, an exception is thrown, since this regressor does not yet
            have smultivariate capability.
        input_checks: boolean
            whether to check the X parameter
        Returns
        -------
        predictions : 1d numpy array
            array of predictions of each instance
        """
        check_is_fitted(self)

        X = check_and_clean_data(X, input_checks=input_checks)

        y_pred = self.model.predict(X, **kwargs)

        if y_pred.ndim == 1:
            y_pred.ravel()
        return y_pred

    def save_trained_model(self):
        save_trained_model(
            self.model, self.model_save_directory, self.model_name
        )
