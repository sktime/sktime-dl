# Utility functions for data handling

__author__ = "James Large"

import pandas as pd
from sktime.utils.data_container import tabularise, nested_to_3d_numpy
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


def check_and_clean_validation_data(validation_X, validation_y, label_encoder,
                                    onehot_encoder, input_checks=True):
    x = validation_X is not None
    y = validation_y is not None

    if x and y:  # both present
        validation_X = check_and_clean_data(validation_X, validation_y,
                                            input_checks=input_checks)

        validation_y_onehot = label_encoder.transform(validation_y)
        validation_y_onehot = validation_y_onehot.reshape(len(validation_y_onehot), 1)
        validation_y_onehot = onehot_encoder.fit_transform(
            validation_y_onehot)

        validation_data = (validation_X, validation_y_onehot)
    elif not x and not y:  # both missing
        validation_data = None
    elif x and not y:  # y missing
        raise ValueError(
            'Given X data for validation, but y labels are missing')
    else:  # x missing
        raise ValueError(
            'Given y labels for validation, but x data are missing')

    return validation_data


def _is_nested_dataframe(X):
    return isinstance(X.iloc[0, 0], pd.Series)


def _univariate_nested_df_to_array(X):
    return tabularise(X, return_array=True)


def _univariate_df_to_array(X):
    return X.to_numpy()


def _multivariate_nested_df_to_array(X):
    X = nested_to_3d_numpy(X)

    # go from [n][d][m] to [n][m][d]
    return X.transpose(0, 2, 1)
