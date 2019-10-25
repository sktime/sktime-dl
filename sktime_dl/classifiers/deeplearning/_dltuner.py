# A basic tuning framework for the deep learning classifiers
# Defaults to a grid search with 5-fold crossvalidation over the param_grid given for the
# specified base_model
#
# Example inputs:
#   base_model=CNNClassifier(),
#   param_grid=dict(
#        kernel_size=[3, 7],
#        avg_pool_size=[2, 3],
#        nb_conv_layers=[1, 2],
#   ),
#
# TODO provide example param_grids for each deep learner

__author__ = "James Large"

import numpy as np

from sktime_dl.classifiers.deeplearning._base import BaseDeepClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


class TunedDeepLearningClassifier(BaseDeepClassifier):

    def __init__(self,
                 base_model,
                 param_grid,
                 n_jobs=1,
                 search_method='grid',
                 cv_folds=5,
                 random_seed=0,
                 verbose=False,
                 model_save_directory=None):

        self.verbose = verbose
        self.model_save_directory = model_save_directory

        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)

        self.base_model = base_model

        # search parameters
        self.param_grid = param_grid
        self.cv_folds = cv_folds
        self.search_method = search_method
        self.n_jobs = n_jobs

        # search results (computed in fit)
        self.grid_history = None
        self.grid = None
        self.model = None  # the best _keras model_, not the sktime classifier object
        self.tuned_params = None

    def build_model(self, input_shape, nb_classes, **kwargs):
        if self.tuned_params is not None:
            return self.base_model.build_model(input_shape, nb_classes, **kwargs)
        else:
            return self.base_model.build_model(input_shape, nb_classes, self.tuned_params)

    def fit(self, X, y, **kwargs):
        """
        Searches the best parameters for and fits classifier on the training set (X, y)
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
        if self.search_method is 'grid':
            self.grid = GridSearchCV(estimator=self.base_model,
                                     param_grid=self.param_grid,
                                     cv=self.cv_folds,
                                     n_jobs=self.n_jobs)
        elif self.search_method is 'random':
            self.grid = RandomizedSearchCV(estimator=self.base_model,
                                           param_grid=self.param_grid,
                                           cv=self.cv_folds,
                                           n_jobs=self.n_jobs)
        else:
            # todo expand, give options etc
            raise Exception('Unrecognised search method provided: {}'.format(self.search_method))

        self.grid_history = self.grid.fit(X, y, refit=True)
        self.model = self.grid.best_estimator_.model
        self.tuned_params = self.grid.best_params_

        # copying data-wrangling info up
        self.label_encoder = self.grid.best_estimator_.label_encoder  #
        self.classes_ = self.grid.best_estimator_.classes_
        self.nb_classes = self.grid.best_estimator_.nb_classes

        if self.verbose:
            self.print_search_summary()

        self.save_trained_model()

        return self

    def get_tuned_model(self):
        return self.model

    def get_tuned_params(self):
        return self.tuned_params

    def print_search_summary(self):
        print("Best: %f using %s" % (self.grid_history.best_score_, self.grid_history.best_params_))
        means = self.grid_history.cv_results_['mean_test_score']
        stds = self.grid_history.cv_results_['std_test_score']
        params = self.grid_history.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
