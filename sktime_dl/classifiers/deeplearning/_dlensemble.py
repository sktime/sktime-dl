# This may be refactored to use standard scikit-learn ensemble mechanisms in the future, currently somewhat bespoke
# for speed of implementation
#
# Ensembles homogeneous randomly-initialised networks with otherwise the same architectures/parameters
#
# Concept originally proposed by:
#
# @article{fawaz2019deep,
#   title={Deep neural network ensembles for time series classification},
#   author={Fawaz, H Ismail and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, P},
#   journal={arXiv preprint arXiv:1903.06602},
#   year={2019}
# }

__author__ = "James Large"

import numpy as np
import pandas as pd
import os

from sklearn.utils.multiclass import class_distribution

from sktime.classifiers.base import BaseClassifier


class DeepLearnerEnsembleClassifier(BaseClassifier):

    def __init__(self, res_path, dataset_name, random_seed=0, verbose=False, nb_iterations=5,
                 network_name='inception'):
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

        # check if binary classification
        if self.y_pred.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            self.y_pred = np.hstack([1 - self.y_pred, self.y_pred])

    def predict_proba(self, X, **kwargs):
        return self.y_pred


# # This may be refactored to use standard scikit-learn ensemble mechanisms in the future, currently somewhat bespoke
# # for speed of implementation
# #
# # Ensembles homogeneous randomly-initialised networks with otherwise the same architectures/parameters
# #
# # Concept originally proposed by:
# #
# # @article{fawaz2019deep,
# #   title={Deep neural network ensembles for time series classification},
# #   author={Fawaz, H Ismail and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, P},
# #   journal={arXiv preprint arXiv:1903.06602},
# #   year={2019}
# # }
#
# __author__ = "James Large"
#
# import numpy as np
# import pandas as pd
#
# from sktime.classifiers.base import BaseClassifier
#
# class DeepLearnerEnsembleClassifier(BaseClassifier):
#
#     def __init__(self, random_seed=0, verbose=False, nb_iterations=5,
#                  network_name='inception'):
#         self.network_name = network_name
#         self.nb_iterations = nb_iterations
#         self.verbose = verbose
#
#         # calced in fit
#         self.classes_ = None
#         self.nb_classes = -1
#         self.models = []
#
#         self.random_seed = random_seed
#         self.random_state = np.random.RandomState(self.random_seed)
#
#     def construct_network(self, model_name, random_seed, verbose=False):
#         model = None
#
#         if model_name == 'CNNClassifier':
#             from ._cnn import CNNClassifier
#             model = CNNClassifier(random_seed, verbose)
#         elif model_name == 'EncoderClassifier':
#             from ._encoder import EncoderClassifier
#             model = EncoderClassifier(random_seed, verbose)
#         elif model_name == 'FCNClassifier':
#             from ._fcn import FCNClassifier
#             model = FCNClassifier(random_seed, verbose)
#         elif model_name == 'InceptionTimeClassifier':
#             from ._inceptiontime import InceptionTimeClassifier
#             model = InceptionTimeClassifier(random_seed, verbose)
#         elif model_name == 'MCDCNNClassifier':
#             from ._mcdcnn import MCDCNNClassifier
#             model = MCDCNNClassifier(random_seed, verbose)
#         elif model_name == 'MCNNClassifier':
#             from ._mcnn import MCNNClassifier
#             model = MCNNClassifier(random_seed, verbose)
#         elif model_name == 'MLPClassifier':
#             from ._mlp import MLPClassifier
#             model = MLPClassifier(random_seed, verbose)
#         elif model_name == 'ResNetClassifier':
#             from ._resnet import ResNetClassifier
#             model = ResNetClassifier(random_seed, verbose)
#         elif model_name == 'TLENETClassifier':
#             from ._tlenet import TLENETClassifier
#             model = TLENETClassifier(random_seed, verbose)
#         elif model_name == 'TWIESNClassifier':
#             from ._twiesn import TWIESNClassifier
#             model = TWIESNClassifier(random_seed, verbose)
#         else:
#             raise Exception('Unrecognised network requested to ensemble over, ' + model_name)
#
#         return model
#
#     def fit(self, X, y, **kwargs):
#
#         # data validation shall be performed in individual network fits
#
#         for itr in range(self.nb_iterations):
#             # each construction shall have a different random initialisation
#             network = self.construct_network(self.network_name, self.random_seed + itr, self.verbose)
#             network.fit(X, y)
#             self.models.append(network)
#
#             if itr == 0:
#                 self.classes_ = network.classes_
#                 self.nb_classes = network.nb_classes
#
#     def predict_proba(self, X, **kwargs):
#
#         if isinstance(X, pd.DataFrame):
#             if X.shape[1] > 1 or not isinstance(X.iloc[0, 0], pd.Series):
#                 raise TypeError(
#                     "Input should either be a 2d numpy array, or a pandas dataframe with a single column of Series objects (networks cannot yet handle multivariate problems")
#             else:
#                 X = np.asarray([a.values for a in X.iloc[:, 0]])
#
#         if len(X.shape) == 2:
#             # add a dimension to make it multivariate with one dimension
#             X = X.reshape((X.shape[0], X.shape[1], 1))
#
#         probs = np.zeros(self.nb_classes)
#
#         for model in self.models:
#             probs = probs + model.predict(X, **kwargs)
#
#         # check if binary classification
#         if probs.shape[1] == 1:
#             # first column is probability of class 0 and second is of class 1
#             probs = np.hstack([1 - probs, probs])
#         return probs
