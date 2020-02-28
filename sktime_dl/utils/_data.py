import pandas as pd
import numpy as np

from sktime.utils.data_container import tabularise

from sktime.utils.validation.supervised import validate_X, validate_X_y

def check_and_clean_data(X, y=None, input_checks=True):
    if input_checks:
        if y is None:
            validate_X(X)
        else:
            validate_X_y(X, y)

    # want data in form: [instances = n][timepoints = m][dimensions = d]
    if isinstance(X, pd.DataFrame):
        if X.shape[1] > 1 and _is_nested_dataframe(X):
            # we have multiple columns, AND each cell contains a series, so this is a multidimensional problem
            X = _multivariate_df_to_array(X)
        else:
            # we either have a single column containing a series, or multiple columns each containing a primitive
            # in either case, treat this as a univariate problem
            X = _univariate_df_to_array(X)

    if len(X.shape) == 2:
        # add a dimension to make it multivariate with one dimension
        X = X.reshape(X.shape[0], X.shape[1], 1)  # go from [n][m] to [n][m][d=1]

    return X


def _is_nested_dataframe(X):
    return isinstance(X.iloc[0, 0], pd.Series)

def _univariate_df_to_array(X):
    return tabularise(X, return_array=True)

def _multivariate_df_to_array(X):
    # tabularise at time of checking will not keep multivariate dimensions separate, e.g. [n][m][d] becomes [n][m*d]
    # todo investigate incorporating the reshaping into the data extraction instead of this 2-stage process
    X = np.array([[X.iloc[r, c].values for c in range(len(X.columns))] for r in range(len(X))])
    return X.reshape(X.shape[0], X.shape[2], X.shape[1])  # go from [n][d][m] to [n][m][d]