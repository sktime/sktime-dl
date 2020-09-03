__author__ = "James Large"

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sktime_dl.deeplearning import CNNClassifier
from sktime_dl.deeplearning.base.estimators import BaseDeepClassifier


class TunedDeepLearningClassifier(BaseDeepClassifier):
    """
    A basic tuning framework for the deep learning classifiers Defaults to a
    grid search with 5-fold crossvalidation over the param_grid given for
    the specified base_model

    TODO provide example param_grids for each deep learner
    """

    def __init__(
            self,
            base_model=CNNClassifier(),
            param_grid=dict(
                kernel_size=[3, 7], avg_pool_size=[2, 3],
                nb_conv_layers=[1, 2],
            ),
            search_method="grid",
            cv_folds=5,
            random_state=0,
            verbose=False,
            model_name=None,
            model_save_directory=None,
    ):
        """
        :param base_model: an implementation of BaseDeepLearner, the model
        to tune :param param_grid: dict, parameter names corresponding to
        parameters of the base_model, mapped to values to search over :param
        search_method: string out of ['grid', 'random'], how to search over
        the param_grid :param cv_folds: int, number of cross validation
        folds to use in evaluation of each parameter set :param random_state:
        int, seed to any needed random actions :param verbose: boolean,
        whether to output extra information :param model_name: string,
        the name of this model for printing and file writing purposes. if
        None, will default to 'tuned_' + base_model.model_name :param
        model_save_directory: string, if not None; location to save the
        tuned, trained keras model in hdf5 format
        """

        self.verbose = verbose

        if model_name is None:
            self.model_name = "tuned_" + base_model.model_name
        else:
            self.model_name = model_name

        self.model_save_directory = model_save_directory

        self.random_state = random_state
        self.random_state = np.random.RandomState(self.random_state)
        self._is_fitted = False

        self.base_model = base_model

        # search parameters
        self.param_grid = param_grid
        self.cv_folds = cv_folds
        self.search_method = search_method
        self.n_jobs = 1  # assuming networks themselves are threaded/on gpu,
        # not providing this option for now

        # search results (computed in fit)
        self.grid_history = None
        self.grid = None
        self.model = (
            None  # the best _keras model_, not the sktime classifier object
        )
        self.tuned_params = None

    def build_model(self, input_shape, nb_classes, **kwargs):
        if self.tuned_params is not None:
            return self.base_model.build_model(
                input_shape, nb_classes, **kwargs
            )
        else:
            return self.base_model.build_model(
                input_shape, nb_classes, self.tuned_params
            )

    def fit(self, X, y, **kwargs):
        """
        Searches the best parameters for and fits classifier on the training
        set (X, y)
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame is passed,
            column 0 is extracted.
        y : array-like, shape = [n_instances]
            The class labels.
        input_checks: boolean
            whether to check the X and y parameters
        Returns
        -------
        self : object
        """
        if self.search_method == "grid":
            self.grid = GridSearchCV(
                estimator=self.base_model,
                param_grid=self.param_grid,
                refit=True,
                cv=self.cv_folds,
                n_jobs=self.n_jobs,
            )
        elif self.search_method == "random":
            self.grid = RandomizedSearchCV(
                estimator=self.base_model,
                param_distributions=self.param_grid,
                refit=True,
                cv=self.cv_folds,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        else:
            # todo expand, give options etc
            raise Exception(
                "Unrecognised search method provided: {}".format(
                    self.search_method
                )
            )

        self.grid_history = self.grid.fit(X, y)
        self.model = self.grid.best_estimator_.model
        self.tuned_params = self.grid.best_params_

        # copying data-wrangling info up
        self.label_encoder = self.grid.best_estimator_.label_encoder
        self.classes_ = self.grid.best_estimator_.classes_
        self.nb_classes = self.grid.best_estimator_.nb_classes

        if self.verbose:
            self.print_search_summary()

        self.save_trained_model()
        self._is_fitted = True

        return self

    def get_tuned_model(self):
        return self.model

    def get_tuned_params(self):
        return self.tuned_params

    def print_search_summary(self):
        print(
            "Best: %f using %s"
            % (self.grid_history.best_score_, self.grid_history.best_params_)
        )
        means = self.grid_history.cv_results_["mean_test_score"]
        stds = self.grid_history.cv_results_["std_test_score"]
        params = self.grid_history.cv_results_["params"]
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
