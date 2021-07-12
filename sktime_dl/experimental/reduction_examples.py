import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series
from sktime_dl.regression import CNNRegressor
from sktime_dl.classification import CNNClassifier
from sktime.forecasting.naive import NaiveForecaster
from sktime.classification.interval_based import RandomIntervalSpectralForest

def forecasting_example():
    name = "C:\\Users\\Tony\\OneDrive - University of East Anglia\\Research\\Alex " \
           "Mcgregor Grant\\randomNoise.csv"


    y = pd.read_csv(name, index_col=0, squeeze=True, dtype={1: np.float})
    forecast_horizon = np.arange(1, 2)
    forecaster = NaiveForecaster(strategy="last")
    forecaster.fit(y)
    y_pred = forecaster.predict(forecast_horizon)
    print("Next predicted value = ",y_pred)
    # https://github.com/alan-turing-institute/sktime/blob/main/examples/01_forecasting.ipynb
    #Reduce to a regression problem through windowing.
    ##Transform forecasting into regression

    np_y = y.to_numpy()
    v = sliding_window_view(y, 100)
    print("Window shape =",v.shape)
    v_3d = np.expand_dims(v, axis=1)
    print("Window shape =",v.shape)
    print(v_3d.shape)
    z = v[:,2]
    print(z.shape)
    regressor = CNNRegressor()
    classifier = CNNClassifier()
    regressor.fit(v_3d,z)
    p = regressor.predict(v_3d)
    #print(p)
    d = np.array([0.0])
    c = np.digitize(z,d)
    classifier = RandomIntervalSpectralForest()
    classifier.fit(v_3d,c)
    cls = classifier.predict(v_3d)
    print(cls)

if __name__ == "__main__":
    forecasting_example()


