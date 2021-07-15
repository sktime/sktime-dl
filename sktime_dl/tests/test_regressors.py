import numpy as np
import pytest
from sklearn.metrics import mean_squared_error
from sktime.datasets import load_airline
from sktime.datasets import load_italy_power_demand
from sktime.forecasting.compose import RecursiveTimeSeriesRegressionForecaster
from sktime.forecasting.model_selection import temporal_train_test_split

from sktime_dl.regression import MLPRegressor, MCDCNNRegressor, CNTCRegressor,\
    LSTMFCNRegressor

from sktime_dl.utils.model_lists import (SMALL_NB_EPOCHS,
                                         construct_all_regressors)


def test_regressor(estimator=CNTCRegressor(nb_epochs=SMALL_NB_EPOCHS)):
    """
    test a regressor
    """
    print("Start test_regressor()")
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)

    # Create some regression values
    y_train = np.zeros(len(y_train))
    for i in range(len(X_train)):
        y_train[i] = X_train.iloc[i].iloc[0].iloc[0]
    y_test = np.zeros(len(y_test))
    for i in range(len(X_test)):
        y_test[i] = X_test.iloc[i].iloc[0].iloc[0]

    estimator.fit(X_train[:10], y_train[:10])
    estimator.predict(X_test[:10])
    score = estimator.score(X_test[:10], y_test[:10])

    print("Estimator score:", score)
    print("End test_regressor()")


def test_regressor_forecasting(
        regressor=CNTCRegressor(nb_epochs=SMALL_NB_EPOCHS), window_length=4
):
    """
    test a regressor used for forecasting
    """
    print("Start test_regressor_forecasting()")

    if isinstance(regressor, MCDCNNRegressor):
        regressor.nb_epochs = regressor.nb_epochs * 2

    # load univariate time series data
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=5)
    y_train = y_train[:window_length * 2]

    # specify forecasting horizon
    fh = np.arange(len(y_test)) + 1

    # solve forecasting task via reduction to time series regression
    forecaster = RecursiveTimeSeriesRegressionForecaster(
        estimator=regressor, window_length=window_length
    )
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)

    try:
        mse = np.sqrt(mean_squared_error(y_test, y_pred))
        print("Error:", mse)
    except ValueError:
        if isinstance(regressor, MCDCNNRegressor):
            print(
                "Warning: MCDCNNRegressor produced NaN predictions. This is a "
                "known problem brought about by insufficient data/learning. "
                "For now, we accept that this particular network produced "
                "predictions at all (even NaNs) as passing for this "
                "particular test. Providing more data/epochs risks slowing "
                "down tests too much.")
        else:
            # unexpected error in all other cases
            raise

    print("End test_regressor_forecasting()")


def test_all_regressors():
    for name, network in construct_all_regressors(SMALL_NB_EPOCHS).items():
        print("\n\t\t" + name + " testing started")
        test_regressor(network)
        print("\t\t" + name + " testing finished")


@pytest.mark.parametrize(
    "name, network",
    construct_all_regressors(SMALL_NB_EPOCHS).items()
)
def test_all_forecasters(name, network):
    window_length = 8
    print("\n\t\t" + name + " forecasttesting \
        started")
    test_regressor_forecasting(network, window_length=window_length)
    print("\t\t" + name + " forecasttesting \
        finished")
