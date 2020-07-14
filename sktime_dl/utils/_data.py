# Utility functions for data handling

__author__ = "James Large"

import numpy as np
import pandas as pd
from sktime.utils.data_container import tabularise
from sktime.utils.validation.series_as_features import check_X, check_X_y


def check_and_clean_data(X, y=None, input_checks=True):
    if input_checks:
        if y is None:
            check_X(X)
        else:
            check_X_y(X, y)

    # want data in form: [instances = n][timepoints = m][dimensions = d]
    if isinstance(X, pd.DataFrame):
        if _is_nested_dataframe(X):
            if X.shape[1] > 1:
                # we have multiple columns, AND each cell contains a series,
                # so this is a multidimensional problem
                X = _multivariate_nested_df_to_array(X)
            else:
                # we have a single column containing a series, treat this as
                # a univariate problem
                X = _univariate_nested_df_to_array(X)
        else:
            # we have multiple columns each containing a primitive, treat as
            # univariate series
            X = _univariate_df_to_array(X)

    if len(X.shape) == 2:
        # add a dimension to make it multivariate with one dimension
        X = X.reshape(
            X.shape[0], X.shape[1], 1
        )  # go from [n][m] to [n][m][d=1]

    return X


def _is_nested_dataframe(X):
    return isinstance(X.iloc[0, 0], pd.Series)


def _univariate_nested_df_to_array(X):
    return tabularise(X, return_array=True)


def _univariate_df_to_array(X):
    return X.to_numpy()


def _multivariate_nested_df_to_array(X):
    # tabularise at time of writing will not keep multivariate dimensions
    # separate, e.g. [n][m][d] becomes [n][m*d]
    # todo investigate incorporating the reshaping into the data extraction
    #  instead of this 2-stage process
    X = np.array(
        [
            [X.iloc[r, c].values for c in range(len(X.columns))]
            for r in range(len(X))
        ]
    )
    return X.reshape(
        X.shape[0], X.shape[2], X.shape[1]
    )  # go from [n][d][m] to [n][m][d]
