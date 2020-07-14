# Model utility functions

__author__ = "Withington, James Large"

from inspect import isclass
from pathlib import Path

from sktime.exceptions import NotFittedError


def save_trained_model(
        model, model_save_directory, model_name, save_format="h5"
):
    """
    Saves the model to an HDF file.

    Saved models can be reinstantiated via `keras.models.load_model`.

    Parameters
    ----------
    save_format: string
        'h5'. Defaults to 'h5' currently but future releases
        will default to 'tf', the TensorFlow SavedModel format.
    """
    if save_format != "h5":
        raise ValueError(
            "save_format must be 'h5'. This is the only format "
            "currently supported."
        )
    if model_save_directory is not None:
        if model_name is None:
            file_name = "trained_model.hdf5"
        else:
            file_name = model_name + ".hdf5"
        path = Path(model_save_directory) / file_name
        model.save(
            path
        )  # Add save_format here upon migration from keras to tf.keras


def check_is_fitted(estimator, msg=None):
    """Perform is_fitted validation for estimator.

    Adapted from sklearn.utils.validation.check_is_fitted

    Checks if the estimator is fitted by verifying the presence and
    positivity of self.is_fitted_

    Parameters
    ----------
    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    Returns
    -------
    None

    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not hasattr(estimator, "_is_fitted") or not estimator.is_fitted:
        raise NotFittedError(msg % {"name": type(estimator).__name__})
