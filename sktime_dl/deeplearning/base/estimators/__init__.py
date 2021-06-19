# -*- coding: utf-8 -*-

__all__ = [
    "BaseDeepNetwork",
    "BaseDeepRegressor",
    "BaseDeepClassifier"
]

from sktime_dl.classification import BaseDeepClassifier
from sktime_dl.networks._network import BaseDeepNetwork
from sktime_dl.regression._regressor import BaseDeepRegressor
