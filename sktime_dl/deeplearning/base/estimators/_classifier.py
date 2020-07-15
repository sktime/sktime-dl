# Base class for the Keras neural network classifiers adapted from Fawaz et. al
# https://github.com/hfawaz/dl-4-tsc

__author__ = "James Large, Aaron Bostrom"

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# from sktime.classifiers.base import BaseClassifier
from sktime.classification.base import BaseClassifier

from sktime_dl.utils import check_and_clean_data
from sktime_dl.utils import check_is_fitted
from sktime_dl.utils import save_trained_model


class BaseDeepClassifier(BaseClassifier):
    def __init__(self, model_name=None, model_save_directory=None):
        self.classes_ = None
        self.nb_classes = None
        self.model_save_directory = model_save_directory
        self.model = None
        self.model_name = model_name

    def build_model(self, input_shape, nb_classes, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for
        training
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        nb_classes: int
            The number of classes, which shall become the size of the output
            layer
        Returns
        -------
        output : a compiled Keras Model
        """
        raise NotImplementedError("this is an abstract method")

    def predict_proba(self, X, input_checks=True, **kwargs):
        """
        Find probability estimates for each class for all cases in X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.
            If a Pandas data frame is passed (sktime format)
            If a Pandas data frame is passed, a check is performed that it
            only has one column.
            If not, an exception is thrown, since this classifier does not
            yet have
            multivariate capability.
        input_checks: boolean
            whether to check the X parameter
        Returns
        -------
        output : array of shape = [n_instances, n_classes] of probabilities
        """
        check_is_fitted(self)

        X = check_and_clean_data(X, input_checks=input_checks)

        probs = self.model.predict(X, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])

        return probs

    def save_trained_model(self):
        save_trained_model(
            self.model, self.model_save_directory, self.model_name
        )

    def convert_y(self, y):
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse=False, categories="auto")
        # categories='auto' to get rid of FutureWarning

        y = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.nb_classes = len(self.classes_)

        y = y.reshape(len(y), 1)
        y = self.onehot_encoder.fit_transform(y)

        return y
