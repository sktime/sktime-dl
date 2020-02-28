import pandas as pd
import numpy as np

from sktime.utils.validation.supervised import validate_X, validate_X_y

def check_and_clean_data(X, y=None, input_checks=True):
    if input_checks:
        if y is None:
            validate_X(X)
        else:
            validate_X_y(X, y)

    from sktime.utils.data_container import tabularise

    print(X.shape, X.iloc[0 ,0].shape)
    X = tabularise(X, return_array=True)
    print(X.shape)

    # want data in form: [instances = n][timepoints = m][dimensions = d]
    if isinstance(X, pd.DataFrame):
        if X.shape[1] > 1:
            # we have multiple columns
            # if cells contain series, this is a multidimensional problem
            # else if cells contain single values, this is a univariate problem with values long columns
            #      this situation can happen with e.g. forecasting-reduced-to-regression strategies
            if isinstance(X.iloc[0, 0], pd.Series):
                # todo investigate incorporating the reshaping into the data extraction instead of this 2-stage process
                X = np.array([[X.iloc[r, c].values for c in range(len(X.columns))] for r in range(len(X))])
                X = X.reshape(X.shape[0], X.shape[2], X.shape[1])  # go from [n][d][m] to [n][m][d]
            else:
                X = X.to_numpy()
        else:
            X = np.asarray([a.values for a in X.iloc[:, 0]])

    if len(X.shape) == 2:
        # add a dimension to make it multivariate with one dimension
        X = X.reshape(X.shape[0], X.shape[1], 1)  # go from [n][m] to [n][m][d=1]

    return X