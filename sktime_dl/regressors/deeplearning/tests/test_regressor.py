import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sktime.highlevel.tasks import ForecastingTask
from sktime.highlevel.strategies import Forecasting2TSRReductionStrategy
from sktime.datasets import load_shampoo_sales
from sktime.transformers.compose import Tabulariser
from sktime.pipeline import Pipeline
from sktime.datasets import load_italy_power_demand 

from sktime_dl.classifiers.deeplearning import CNNClassifier
from sktime_dl.regressors.deeplearning import CNNRegressor



def test_regressor():
    '''
    Test 
    '''

    print("Start test_regressor()")
    X_train, y_train = load_italy_power_demand(split='TRAIN', return_X_y=True)
    X_test, y_test = load_italy_power_demand(split='TEST', return_X_y=True)

    y_train = y_train.astype(int)
    y_train = np.multiply(y_train, 5.1)
    y_test = y_test.astype(int)
    y_test = np.multiply(y_test, 5.1)
    print(y_train[:5])


    # CNNRegressor
    estimator = CNNRegressor()

    print(X_train.shape)
    estimator.fit(X_train[:10], y_train[:10], input_checks=True, verbose=2)

    y_pred = estimator.predict(X_test[:10])
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ CNNRegressor y_pred:', y_pred)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ CNNRegressor y_test:', y_test[:10])
    print('@@@@@@@@@@@@@@@@@@@@@@@@@ y_pred.shape:', y_pred.shape)
    
    print('Estimator score:', estimator.score(X_test[:10], y_test[:10]))

    #assert(1==0)
    print("End test_regressor()")


def test_regressor_forecasting():
    '''
    Test 
    '''

    print("Start test_regressor_forecasting()")

    # get data into expected nested format
    shampoo = load_shampoo_sales(return_y_as_dataframe=True)
    train = pd.DataFrame(pd.Series([shampoo.iloc[0, 0].iloc[:24]]), columns=shampoo.columns)
    update = pd.DataFrame(pd.Series([shampoo.iloc[0, 0].iloc[:30]]), columns=shampoo.columns)
    test = pd.DataFrame(pd.Series([shampoo.iloc[0, 0].iloc[30:]]), columns=shampoo.columns)

    # define simple time-series regressor using time-series as features
    steps = [
        ('tabularise', Tabulariser()),
        ('clf', RandomForestRegressor(n_estimators=100))
    ]
    estimator = Pipeline(steps)

    task = ForecastingTask(target='ShampooSales', fh=[1, 2], metadata=train)

    s = Forecasting2TSRReductionStrategy(estimator=estimator, window_length=9)
    s.fit(task, train)
    y_pred = s.predict()
    print('RandomForestRegressor y_pred:', y_pred)
    # Compare the prediction to the test data
    test = update.iloc[0, 0][y_pred.index]
    print('RandomForestRegressor error:', np.sqrt(mean_squared_error(test, y_pred)))

    # CNNRegressor
    steps = [
        ('tabularise', Tabulariser()),
        ('clf', CNNRegressor())
    ]
    estimator = Pipeline(steps)

    task = ForecastingTask(target='ShampooSales', fh=[1, 2], metadata=train)

    s = Forecasting2TSRReductionStrategy(estimator=estimator, window_length=9)
    s.fit(task, train)
    y_pred = s.predict()
    print('CNNRegressor y_pred:', y_pred)

    # Compare the prediction to the test data
    test = update.iloc[0, 0][y_pred.index]
    print('++++++++++++++++++++++++++++++++CNNRegressor error:', np.sqrt(mean_squared_error(test, y_pred)))

    #assert(1==0)
    print("End test_regressor_forecasting()")
