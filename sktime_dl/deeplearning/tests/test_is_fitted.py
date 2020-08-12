import numpy as np
import pytest
from sktime.exceptions import NotFittedError
from sktime.datasets import load_italy_power_demand
from sktime.regression.base import BaseRegressor

from sktime_dl.deeplearning import CNNClassifier
from sktime_dl.utils.model_lists import SMALL_NB_EPOCHS
from sktime_dl.utils.model_lists import construct_all_classifiers
from sktime_dl.utils.model_lists import construct_all_regressors


def test_is_fitted(network=CNNClassifier()):
    """
    testing that the networks correctly recognise when they are not fitted
    """

    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)

    if isinstance(network, BaseRegressor):
        # Create some regression values, taken from test_regressor
        y_train = np.zeros(len(y_train))
        for i in range(len(X_train)):
            y_train[i] = X_train.iloc[i].iloc[0].iloc[0]

    # try to predict without fitting: SHOULD fail
    with pytest.raises(NotFittedError):
        network.predict(X_train[:10])


def test_all_networks():
    networks = {
        **construct_all_classifiers(SMALL_NB_EPOCHS),
        **construct_all_regressors(SMALL_NB_EPOCHS),
    }

    for name, network in networks.items():
        print("\n\t\t" + name + " is_fitted testing started")
        test_is_fitted(network)
        print("\t\t" + name + " is_fitted testing finished")


if __name__ == "__main__":
    test_all_networks()
