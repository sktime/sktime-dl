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
from sktime_dl.regressors.deeplearning import TempCNNRegressor

def test_regressor():
    '''
    test a regressor
    '''

    print("Start test_regressor() ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    X_train, y_train = load_italy_power_demand(split='TRAIN', return_X_y=True)
    X_test, y_test = load_italy_power_demand(split='TEST', return_X_y=True)

    # Create some regression values
    y_train = np.multiply(y_train.astype(int), 5.1)
    y_test = np.multiply(y_test.astype(int), 5.1)

    estimator = CNNRegressor()

    if False:
        estimator.fit(X_train[:10], y_train[:10])
        y_pred = estimator.predict(X_test[:10])
        score = estimator.score(X_test[:10], y_test[:10])
    else:
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        score = estimator.score(X_test, y_test)

    print('Estimator score:', score)
    print(y_pred)
    assert(False)
    #assert(score > -3 and score < -2)
    print("End test_regressor()")


def test_regressor_forecasting():
    '''
    test a regressor used for forecasting 
    '''

    print("Start test_regressor_forecasting() ++++++++++++++++++++++++++++++++")

    # get data into expected nested format
    
    shampoo = load_shampoo_sales(return_y_as_dataframe=True)
    shampoo_train = pd.DataFrame(pd.Series([shampoo.iloc[0, 0].iloc[:24]]), columns=shampoo.columns)
    shampoo_update = pd.DataFrame(pd.Series([shampoo.iloc[0, 0].iloc[:30]]), columns=shampoo.columns)
    #shampoo_test = pd.DataFrame(pd.Series([shampoo.iloc[0, 0].iloc[30:]]), columns=shampoo.columns)

    raw_seq = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
    seq_df = pd.DataFrame(pd.Series(raw_seq))
    seq = pd.DataFrame(pd.Series([pd.Series(seq_df.squeeze())]), columns=['ShampooSales'])
    seq_train = pd.DataFrame(pd.Series([seq.iloc[0, 0].iloc[:12]]), columns=['ShampooSales'])
    seq_update = pd.DataFrame(pd.Series([seq.iloc[0, 0].iloc[12:]]), columns=['ShampooSales'])
    #seq_train = pd.DataFrame(pd.Series([raw_seq[:9]]), columns=['ShampooSales'])
    #seq_update = pd.DataFrame(pd.Series([raw_seq[9:]]), columns=['ShampooSales'])
    
    #print(shampoo_train)
    #print(seq_train)


    # Set the problem to forecast
    train = seq_train
    update = seq_update
    window_len = 9

    #regressor = RandomForestRegressor(n_estimators=2)
    #regressor = CNNRegressor(kernel_size=3, filter_sizes=[16, 32], avg_pool_size=1)
    regressor = TempCNNRegressor()

    # define simple time-series regressor using time-series as features
    steps = [
        ('tabularise', Tabulariser()),
        ('rgs', regressor)
    ]
    estimator = Pipeline(steps)

    task = ForecastingTask(target='ShampooSales', fh=[1], metadata=train)


    s = Forecasting2TSRReductionStrategy(estimator=estimator, 
            window_length=window_len)
    s.fit(task, train)
    y_pred = s.predict()
    print('Prediction y_pred:', y_pred)
    # Compare the prediction to the test data
    test = update.iloc[0, 0][y_pred.index]
    mse = np.sqrt(mean_squared_error(test, y_pred))
    print('Error:', mse)

    assert(False)
    #assert(mse > 360 and mse < 380)
    print("End test_regressor_forecasting()")
