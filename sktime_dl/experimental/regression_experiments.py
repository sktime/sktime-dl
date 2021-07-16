# -*- coding: utf-8 -*-
"""Experiments: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format
todo: Tidy up this file!
"""

import os

import sklearn.preprocessing
import sklearn.utils



os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sys
import time
import numpy as np
import pandas as pd


from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sktime.contrib.experiments import run_experiment
from sktime.datasets.base import load_UCR_UEA_dataset
from sktime.contrib.experiments import write_results_to_uea_format


from sktime_dl.regression import (
    CNNRegressor,
    EncoderClassifier,
    EncoderRegressor,
    FCNClassifier,
    FCNRegressor,
    InceptionTimeClassifier,
    InceptionTimeRegressor,
    LSTMRegressor,
    MCDCNNClassifier,
    MCDCNNRegressor,
    MCNNClassifier,
    MLPClassifier,
    MLPRegressor,
    ResNetClassifier,
    ResNetRegressor,
    SimpleRNNRegressor,
    TLENETClassifier,
    TLENETRegressor,
    TWIESNClassifier,
)

__author__ = ["Tony Bagnall"]

"""Prototype mechanism for testing classifiers on the UCR format. This mirrors the
mechanism used in Java,
https://github.com/TonyBagnall/uea-tsc/tree/master/src/main/java/experiments
but is not yet as engineered. However, if you generate results using the method
recommended here, they can be directly and automatically compared to the results
generated in java

"""

regressor_list = [
    "CNNRegressor",
    "EncoderRegressor",
    "FCNRegressor",
    "InceptionTimeRegressor",
    "LSTMRegressor",
    "MCDCNNRegressor",
    "MLPRegressor",
    "ResNetRegressor",
    "SimpleRNNRegressor",
    "TLENETRegressor",
]

def set_regressor(cls, resampleId=None):
    """Construct a classifier.

    Basic way of creating the classifier to build using the default settings. This
    set up is to help with batch jobs for multiple problems to facilitate easy
    reproducability. You can set up bespoke classifier in many other ways.

    Parameters
    ----------
    cls: String indicating which classifier you want
    resampleId: classifier random seed

    Return
    ------
    A classifier.
    """
    name = cls.lower()
    # Convolutional
    if name == "cnn" or name == "cnnregressor":
        return CNNRegressor(random_state=resampleId)
    elif name == "encode":
        return EncoderRegressor()
    elif name == "fcn":
        return FCNRegressor()
    elif name == "inceptiontime":
        return InceptionTimeRegressor()
    elif name == "mcdcnn":
        return MCDCNNRegressor()
    elif name == "mcnn":
        return MCNNRegressor()
    elif name == "mlp":
        return MLPRegressor()
    elif name == "resnet":
        return ResNetRegressor()
    elif name == "tlenet":
        return TLENETRegressor()
    elif name == "twiesn":
        return TWIESNClassifier()
    else:
        raise Exception("UNKNOWN CLASSIFIER")


