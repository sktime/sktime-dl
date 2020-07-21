import numpy as np
import pytest
from sktime.datasets import load_italy_power_demand
from sktime.regression.base import BaseRegressor

from sktime_dl.deeplearning import MLPClassifier
from sktime_dl.utils.model_lists import SMALL_NB_EPOCHS
from sktime_dl.utils.model_lists import construct_all_classifiers
from sktime_dl.utils.model_lists import construct_all_regressors

from sktime_dl.deeplearning import InceptionTimeClassifier
from sktime_dl.deeplearning import ResNetClassifier


def test_validation(network=MLPClassifier()):
    """
    testing that the networks correctly recognise when they are not fitted
    """

    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)

    X_test = X_train[6:10]
    y_test = y_train[6:10]

    X_train = X_train[:5]
    y_train = y_train[:5]

    if isinstance(network, BaseRegressor):
        # Create some regression values, taken from test_regressor
        y_train = np.zeros(len(y_train))
        for i in range(len(X_train)):
            y_train[i] = X_train.iloc[i].iloc[0].iloc[0]

    network.fit(X_train, y_train, validation_X=X_test, validation_y=y_test)
    hist = network.history.history

    assert ('val_loss' in hist) and ('val_accuracy' in hist)


def test_all_networks():
    # expand to all once implemented in all
    # networks = construct_all_classifiers(
    #     SMALL_NB_EPOCHS
    # ) + construct_all_regressors(SMALL_NB_EPOCHS)
    networks = [
        MLPClassifier(nb_epochs=SMALL_NB_EPOCHS),
        ResNetClassifier(nb_epochs=SMALL_NB_EPOCHS),
        InceptionTimeClassifier(nb_epochs=SMALL_NB_EPOCHS),
    ]

    for network in networks:
        print(
            "\n\t\t"
            + network.__class__.__name__
            + " is_fitted testing started"
        )
        test_validation(network)
        print(
            "\t\t" + network.__class__.__name__ + " is_fitted testing finished"
        )


if __name__ == "__main__":
    test_all_networks()
