import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

from sktime.highlevel.tasks import ForecastingTask
from sktime.highlevel.strategies import Forecasting2TSRReductionStrategy
from sktime.datasets import load_shampoo_sales, load_italy_power_demand
from sktime.transformers.compose import Tabulariser
from sktime.pipeline import Pipeline

from sktime_dl.deeplearning import CNNRegressor
from sktime_dl.deeplearning import EncoderRegressor
from sktime_dl.deeplearning import FCNRegressor
from sktime_dl.deeplearning import MLPRegressor
from sktime_dl.deeplearning import ResNetRegressor
from sktime_dl.deeplearning import TLENETRegressor
from sktime_dl.deeplearning import InceptionTimeRegressor
from sktime_dl.deeplearning import SimpleRNNRegressor
from sktime_dl.deeplearning import LSTMRegressor


def test_regressor(estimator=MLPRegressor(nb_epochs=10)):
    '''
    test a regressor
    '''
    print("Start test_regressor()")
    X_train, y_train = load_italy_power_demand(split='TRAIN', return_X_y=True)
    X_test, y_test = load_italy_power_demand(split='TEST', return_X_y=True)

    # Create some regression values
    y_train = np.zeros(len(y_train))
    for i in range(len(X_train)):
        y_train[i] = X_train.iloc[i].iloc[0].iloc[0]
    y_test = np.zeros(len(y_test))
    for i in range(len(X_test)):
        y_test[i] = X_test.iloc[i].iloc[0].iloc[0]

    estimator.fit(X_train[:10], y_train[:10])
    __ = estimator.predict(X_test[:10])
    score = estimator.score(X_test[:10], y_test[:10])

    print('Estimator score:', score)
    print("End test_regressor()")


def test_regressor_forecasting(estimator=MLPRegressor(nb_epochs=10),
                               window_length=4):
    '''
    test a regressor used for forecasting 
    '''
    print("Start test_regressor_forecasting()")

    # get data into expected nested format
    shampoo = load_shampoo_sales(return_y_as_dataframe=True)
    train = pd.DataFrame(pd.Series([shampoo.iloc[0, 0].iloc[:24]]), columns=shampoo.columns)
    update = pd.DataFrame(pd.Series([shampoo.iloc[0, 0].iloc[:30]]), columns=shampoo.columns)

    # define simple time-series regressor using time-series as features
    steps = [
        ('tabularise', Tabulariser()),
        ('rgs', estimator)
    ]
    estimator = Pipeline(steps)

    task = ForecastingTask(target='ShampooSales', fh=[1], metadata=train)
    strategy = Forecasting2TSRReductionStrategy(estimator=estimator,
                                                window_length=window_length)
    strategy.fit(task, train)
    y_pred = strategy.predict()

    # Compare the prediction to the test data
    test = update.iloc[0, 0][y_pred.index]
    mse = np.sqrt(mean_squared_error(test, y_pred))
    print('Error:', mse)
    print("End test_regressor_forecasting()")


def test_all_regressors():
    networks = [
        CNNRegressor(nb_epochs=10),
        EncoderRegressor(nb_epochs=10),
        FCNRegressor(nb_epochs=10),
        MLPRegressor(nb_epochs=10),
        ResNetRegressor(nb_epochs=10),
        TLENETRegressor(nb_epochs=10),
        InceptionTimeRegressor(nb_epochs=10),
        SimpleRNNRegressor(nb_epochs=3),
        LSTMRegressor(nb_epochs=10)
    ]

    for network in networks:
        print('\n\t\t' + network.__class__.__name__ + ' testing started')
        test_regressor(network)
        print('\t\t' + network.__class__.__name__ + ' testing finished')


def test_all_forecasters():
    window_length = 4
    # [[network, window length]]
    networks = [
        [CNNRegressor(nb_epochs=10, kernel_size=3, avg_pool_size=1)],
        [EncoderRegressor(nb_epochs=10)],
        [FCNRegressor(nb_epochs=10)],
        [MLPRegressor(nb_epochs=10)],
        [ResNetRegressor(nb_epochs=10)],
        [TLENETRegressor(nb_epochs=10), 8],
        [InceptionTimeRegressor(nb_epochs=10)],
        [SimpleRNNRegressor(nb_epochs=3)],
        [LSTMRegressor(nb_epochs=10)]
    ]

    for network in networks:
        print('\n\t\t' + network[0].__class__.__name__ + ' forecast testing started')
        win_len = window_length if len(network) == 1 else network[1]
        test_regressor_forecasting(network[0], window_length=win_len)
        print('\t\t' + network[0].__class__.__name__ + ' forecast testing finished')


if __name__ == "__main__":
    test_all_regressors()
