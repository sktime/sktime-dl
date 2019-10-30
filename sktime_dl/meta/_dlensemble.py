__author__ = "James Large"

import numpy as np
import pandas as pd
import os
import keras
import gc

from keras.models import load_model

from sklearn.utils.multiclass import class_distribution
from sklearn.base import clone

from sktime.classifiers.base import BaseClassifier

from sktime_dl.classifiers.deeplearning import InceptionTimeClassifier


class DeepLearnerEnsembleClassifier(BaseClassifier):
    '''
    Simplified/streamlined class to ensemble over homogeneous network architectures with different random initialisations

    This may be refactored to use standard scikit-learn ensemble mechanisms in the future, currently somewhat bespoke
    for speed of implementation

    Originally proposed by:

    @article{fawaz2019deep,
      title={Deep neural network ensembles for time series classification},
      author={Fawaz, H Ismail and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, P},
      journal={arXiv preprint arXiv:1903.06602},
      year={2019}
    }
    '''

    def __init__(self,
                 base_model=InceptionTimeClassifier(),
                 nb_iterations=5,
                 keep_in_memory=False,

                 random_seed=0,
                 verbose=False,
                 model_name="InceptionTime",
                 model_save_directory=None):
        '''
        :param base_model: an implementation of BaseDeepLearner, the model to ensemble over. 
                                MUST NOT have had fit called on it
        :param nb_iterations: int, the number of models to ensemble over
        :param keep_in_memory: boolean, if True, all models will be kept in memory while fitting/predicting.
                Otherwise, models will be written to/read from file individually while fitting/predicting.
                model_name and model_save_directory must be set in this case
        :param random_seed: int, seed to any needed random actions
        :param verbose: boolean, whether to output extra information
        :param model_name: string, the name of this model for printing and file writing purposes. if None, will default
                to base_model.model_name + '_ensemble'
        :param model_save_directory: string, if not None; location to save the trained BASE MODELS of the ensemble
        '''

        self.verbose = verbose

        if model_name is None:
            self.model_name = base_model.model_name + "_ensemble"
        else:
            self.model_name = model_name

        self.model_save_directory = model_save_directory

        self.base_model = base_model
        self.nb_iterations = nb_iterations
        self.keep_in_memory = keep_in_memory

        # calced in fit
        self.classes_ = None
        self.nb_classes = -1
        self.models = []

        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)

    def construct_model(self, itr):
        model = clone(self.base_model)
        model.random_seed = self.random_seed + itr

        if self.model_save_directory is not None:
            model.model_save_directory = self.model_save_directory

        model.model_name = model.model_name + str(itr)

        return model

    def fit(self, X, y, input_checks=True, **kwargs):
        """
        Build the ensemble constituents on the training set (X, y)
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame is passed, column 0 is extracted.
        y : array-like, shape = [n_instances]
            The class labels.
        input_checks: boolean
            whether to check the X and y parameters
        Returns
        -------
        self : object
        """
        self.models = []

        # data validation shall be performed in individual network fits
        for itr in range(self.nb_iterations):

            # each construction shall have a different random initialisation
            model = self.construct_model(itr)
            model.fit(X, y)

            if itr == 0:
                self.classes_ = model.classes_
                self.nb_classes = model.nb_classes

            if self.keep_in_memory:
                self.models.append(model)
            else:
                self.models.append(model.model_name)

                del model
                gc.collect()
                keras.backend.clear_session()

    def predict_proba(self, X, input_checks=True, **kwargs):
        """
        Find probability estimates for each class for all cases in X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.
            If a Pandas data frame is passed (sktime format)
            If a Pandas data frame is passed, a check is performed that it only has one column.
            If not, an exception is thrown, since this classifier does not yet have
            multivariate capability.
        input_checks: boolean
            whether to check the X parameter
        Returns
        -------
        output : array of shape = [n_instances, n_classes] of probabilities
        """
        X = self.check_and_clean_data(X, input_checks=input_checks)

        probs = np.zeros(self.nb_classes)

        for model in self.models:
            if not self.keep_in_memory:
                model = load_model(self.model_save_directory + model)

            probs = probs + model.predict(X, **kwargs)

            if not self.keep_in_memory:
                del model
                gc.collect()
                keras.backend.clear_session()

        probs = probs / len(self.models)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs


class EnsembleFromFileClassifier(BaseClassifier):
    '''
    A simple utility for post-hoc ensembling over the results of networks that have already been trained and had their results
    saved via sktime.contrib.experiments.py
    '''

    def __init__(self,
                 res_path,
                 dataset_name,
                 nb_iterations=5,
                 network_name='inception',
                 random_seed=0,
                 verbose=False):
        self.network_name = network_name
        self.nb_iterations = nb_iterations
        self.verbose = verbose

        self.res_path = res_path
        self.dataset_name = dataset_name

        # calced in fit
        self.classes_ = None
        self.nb_classes = -1
        self.models = []

        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)

    def load_network_probs(self, network_name, itr, res_path, dataset_name, fold):
        path = os.path.join(res_path, network_name + str(itr), "Predictions", dataset_name,
                            "testFold" + str(fold) + ".csv")
        probs = pd.read_csv(path, engine='python', skiprows=3, header=None)
        return np.asarray(probs)[:, 3:]

    def fit(self, X, y, **kwargs):
        self.nb_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        for itr in range(self.nb_iterations):
            # each construction shall have a different random initialisation
            y_cur = self.load_network_probs(self.network_name, itr, self.res_path, self.dataset_name, self.random_seed)

            if itr == 0:
                self.y_pred = y_cur
            else:
                self.y_pred = self.y_pred + y_cur

        self.y_pred = self.y_pred / self.nb_iterations

        # check if binary classification
        if self.y_pred.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            self.y_pred = np.hstack([1 - self.y_pred, self.y_pred])

        return self

    def predict_proba(self, X, **kwargs):
        return self.y_pred
