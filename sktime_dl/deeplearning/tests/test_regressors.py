import numpy as np
# from sklearn.metrics import mean_squared_error
from sktime.datasets import load_italy_power_demand
# from sktime.datasets import load_airline
# from sktime.forecasting.compose import ReducedRegressionForecaster
# from sktime.forecasting.model_selection import temporal_train_test_split
# from sktime.forecasting.compose import ReducedTimeSeriesRegressionForecaster
from sktime_dl.deeplearning import MLPRegressor
from sktime_dl.utils.model_lists import (SMALL_NB_EPOCHS,
                                         construct_all_regressors)
# from sktime.utils.data_container import tabularize

#


def test_regressor(estimator=MLPRegressor(nb_epochs=SMALL_NB_EPOCHS)):
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


# def test_regressor_forecasting(
#         regressor=MLPRegressor(nb_epochs=SMALL_NB_EPOCHS), window_length=4
# ):
#     """
#     test a regressor used for forecasting
#     """
#     print("Start test_regressor_forecasting()")

#     # get data into expected nested format
#     y = load_airline()
#     y_train, y_test = temporal_train_test_split(y, test_size=36)

#     # y_train = tabularize(y_train, return_array=True)
#     # y_test = tabularize(y_test, return_array=True)

#     # from sklearn.neighbors import KNeighborsRegressor
#     # regressor = KNeighborsRegressor(n_neighbors=1)

#     # define simple time-series regressor using time-series as features
#     forecaster = ReducedRegressionForecaster(
#         regressor=regressor, window_length=window_length
#     )
#     forecaster.fit(y_train)
#     fh = np.arange(len(y_test)) + 1
#     y_pred = forecaster.predict(fh)

#     # Compare the prediction to the test data
#     mse = np.sqrt(mean_squared_error(y_test, y_pred))
#     print("Error:", mse)
#     print("End test_regressor_forecasting()")


def test_all_regressors():
    for network in construct_all_regressors(SMALL_NB_EPOCHS):
        print("\n\t\t" + network.__class__.__name__ + " testing started")
        test_regressor(network)
        print("\t\t" + network.__class__.__name__ + " testing finished")

# def test_all_forecasters():
#     window_length = 8

#     for network in construct_all_regressors(SMALL_NB_EPOCHS):
#         print("\n\t\t" + network.__class__.__name__ + " forecasttesting \
#          started")
#         test_regressor_forecasting(network, window_length=window_length)
#         print("\t\t" + network.__class__.__name__ + " forecasttesting \
#             finished")
