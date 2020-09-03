__author__ = "Aaron Bostrom, James Large"

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as slinalg
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sktime_dl.deeplearning.base.estimators import BaseDeepClassifier
from sktime_dl.utils import check_and_clean_data
from sktime_dl.utils import check_is_fitted
from sklearn.utils import check_random_state


class TWIESNClassifier(BaseDeepClassifier):
    """Time Warping Invariant Echo State Network (TWIESN).

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/twiesn.py

    Network originally defined in:

    @inproceedings{tanisaro2016time, title={Time series classification using
    time warping invariant echo state networks}, author={Tanisaro, Pattreeya
    and Heidemann, Gunther}, booktitle={2016 15th IEEE International
    Conference on Machine Learning and Applications (ICMLA)}, pages={
    831--836}, year={2016}, organization={IEEE} }
    """

    def __init__(
            self,
            rho_s=[0.55, 0.9, 2.0, 5.0],
            alpha=0.1,  # leaky rate
            random_state=0,
            verbose=False,
            model_name="twiesn",
            model_save_directory=None,
    ):
        """
        :param rho_s: array of shape
        :param alpha: float, the leakage rate
        :param random_state: int, seed to any needed random actions
        :param verbose: boolean, whether to output extra information
        :param model_name: string, the name of this model for printing and
         file writing purposes
        :param model_save_directory: string, if not None; location to save
         the trained keras model in hdf5 format
        """
        super(TWIESNClassifier, self).__init__(
            model_name,
            model_save_directory)
        self.rho_s = rho_s
        self.alpha = alpha  # leakage rate

        self.random_state = random_state
        self.verbose = verbose
        self.model_name = model_name
        self.model_save_directory = model_save_directory

        self._is_fitted = False

    def set_hyperparameters(self):
        # hyperparameters
        first_config = {
            "N_x": 250,
            "connect": 0.5,
            "scaleW_in": 1.0,
            "lamda": 0.0,
        }
        second_config = {
            "N_x": 250,
            "connect": 0.5,
            "scaleW_in": 2.0,
            "lamda": 0.05,
        }
        third_config = {
            "N_x": 500,
            "connect": 0.1,
            "scaleW_in": 2.0,
            "lamda": 0.05,
        }
        fourth_config = {
            "N_x": 800,
            "connect": 0.1,
            "scaleW_in": 2.0,
            "lamda": 0.05,
        }
        self.configs = [
            first_config,
            second_config,
            third_config,
            fourth_config,
        ]

    def evaluate_paramset(self, X, y, val_X, val_y, rho, config):

        # param setting is correct.
        self.rho = rho
        self.N_x = config["N_x"]
        self.connect = config["connect"]
        self.scaleW_in = config["scaleW_in"]
        self.lamda = config["lamda"]

        # init transformer based on paras.
        self.init_matrices()

        # transformed X
        X_transformed = self.transform_to_feature_space(X)

        y_new = np.repeat(y, self.T, axis=0)

        ridge_classifier = Ridge(alpha=self.lamda)
        ridge_classifier.fit(X_transformed, y_new)

        # transform Validation and labels
        val_X_transformed = self.transform_to_feature_space(val_X)

        val_preds = ridge_classifier.predict(val_X_transformed)
        val_preds = self.reshape_prediction(val_preds, val_X.shape[0], self.T)

        # calculate validation accuracy
        # argmax the val_y because it is in onehot encoding.
        return accuracy_score(np.argmax(val_y, axis=1), val_preds)

    def fit(self, X, y, input_checks=True, **kwargs):
        """
        Fit the classifier on the training set (X, y)
        ----------
        X : a nested pd.Dataframe, or (if input_checks=False) array-like of
        shape = (n_instances, series_length, n_dimensions)
            The training input samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
        y : array-like, shape = [n_instances]
            The training data class labels.
        input_checks : boolean
            whether to check the X and y parameters
        Returns
        -------
        self : object
        """
        self.random_state = check_random_state(self.random_state)

        self.set_hyperparameters()

        X = check_and_clean_data(X, y, input_checks=input_checks)
        y_onehot = self.convert_y(y)

        # ignore the number of instances, X.shape[0],
        # just want the shape of each instance
        self.input_shape = X.shape[1:]

        self.num_dim = X.shape[2]
        self.T = X.shape[1]

        # FINE TUNE MODEL PARAMS
        # split train to validation set to choose best hyper parameters
        x_train, x_val, y_train, y_val = train_test_split(
            X, y_onehot, test_size=0.2
        )
        self.N = x_train.shape[0]

        # limit the hyperparameter search if dataset is too big
        if x_train.shape[0] > 1000:
            for config in self.configs:
                config["N_x"] = 100
            self.configs = [self.configs[0], self.configs[1], self.configs[2]]

        # search for best hyper parameters
        best_train_acc = -1
        best_rho = -1
        best_config = None
        for idx_config in range(len(self.configs)):
            for rho in self.rho_s:
                train_acc = self.evaluate_paramset(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    rho,
                    self.configs[idx_config],
                )

                # print(train_acc)
                if best_train_acc < train_acc:
                    best_train_acc = train_acc
                    best_rho = rho
                    best_config = self.configs[idx_config]

        self.rho = best_rho
        self.N_x = best_config["N_x"]
        self.connect = best_config["connect"]
        self.scaleW_in = best_config["scaleW_in"]
        self.lamda = best_config["lamda"]

        # init transformer based on paras.
        self.init_matrices()

        # transformed X
        X_transformed = self.transform_to_feature_space(X)

        # transform the corresponding labels
        y_new = np.repeat(y_onehot, self.T, axis=0)

        # create and fit the tuned ridge classifier.
        self.model = Ridge(alpha=self.lamda)
        self.model.fit(X_transformed, y_new)

        self.save_trained_model()
        self._is_fitted = True

        return self

    def predict_proba(self, X, input_checks=True, **kwargs):
        """
        Find probability estimates for each class for all cases in X.
        Parameters
        ----------
        X : a nested pd.Dataframe, or (if input_checks=False) array-like of
        shape = (n_instances, series_length, n_dimensions)
            The training input samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
        input_checks: boolean
            whether to check the X parameter
        Returns
        -------
        output : array of shape = [n_instances, n_classes] of probabilities
        """
        check_is_fitted(self)

        X = check_and_clean_data(X, input_checks=input_checks)

        # transform and predict prodba on the ridge classifier.
        X_transformed = self.transform_to_feature_space(X)
        y_pred = self.model.predict(X_transformed)

        # self.reshape_prediction will give us PREDICTIONS,
        # not DISTRIBUTIONS (even if only one-hot)
        # Computing first 2 lines of that but not the last here

        # reshape so the first axis has the number of instances
        new_y_pred = y_pred.reshape(X.shape[0], X.shape[1], y_pred.shape[-1])
        # average the predictions of instances
        return np.average(new_y_pred, axis=1)

    def init_matrices(self):
        self.W_in = (2.0 * np.random.rand(self.N_x, self.num_dim) - 1.0) / (
                2.0 * self.scaleW_in
        )

        converged = False

        i = 0

        # repeat because could not converge to find eigenvalues
        while not converged:
            i += 1

            # generate sparse, uniformly distributed weights
            self.W = sparse.rand(
                self.N_x, self.N_x, density=self.connect
            ).todense()

            # ensure that the non-zero values are uniformly distributed
            self.W[np.where(self.W > 0)] -= 0.5

            try:
                # get the largest eigenvalue
                eig, _ = slinalg.eigs(self.W, k=1, which="LM")
                converged = True
            except Exception:
                print("not converged ", i)
                continue

        # adjust the spectral radius
        self.W /= np.abs(eig) / self.rho

    def compute_state_matrix(self, x_in):
        # number of instances
        n = x_in.shape[0]
        # the state matrix to be computed
        X_t = np.zeros((n, self.T, self.N_x), dtype=np.float64)
        # previous state matrix
        X_t_1 = np.zeros((n, self.N_x), dtype=np.float64)
        # loop through each time step
        for t in range(self.T):
            # get all the time series data points for the time step t
            curr_in = x_in[:, t, :]
            # calculate the linear activation
            curr_state = np.tanh(
                self.W_in.dot(curr_in.T) + self.W.dot(X_t_1.T)
            ).T
            # apply leakage
            curr_state = (1 - self.alpha) * X_t_1 + self.alpha * curr_state
            # save in previous state
            X_t_1 = curr_state
            # save in state matrix
            X_t[:, t, :] = curr_state

        return X_t

    def transform_to_feature_space(self, X):
        # compute the state matrices which is the new feature space
        state_matrix = self.compute_state_matrix(X)
        # add the input to form the new feature space and transform to
        # the new feature space to be feeded to the classifier
        return np.concatenate((X, state_matrix), axis=2).reshape(
            X.shape[0] * self.T, self.num_dim + self.N_x
        )

    def reshape_prediction(self, y_pred, num_instances, length_series):
        # reshape so the first axis has the number of instances
        new_y_pred = y_pred.reshape(
            num_instances, length_series, y_pred.shape[-1]
        )
        # average the predictions of instances
        new_y_pred = np.average(new_y_pred, axis=1)
        # get the label with maximum prediction over the last label axis
        new_y_pred = np.argmax(new_y_pred, axis=1)
        return new_y_pred
