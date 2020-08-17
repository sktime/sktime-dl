import os

import numpy as np
import pytest
from sktime.datasets import load_italy_power_demand
from sktime.regression.base import BaseRegressor

from sktime_dl.deeplearning import MLPClassifier
from sktime_dl.utils.model_lists import SMALL_NB_EPOCHS
from sktime_dl.utils.model_lists import construct_all_classifiers
from sktime_dl.utils.model_lists import construct_all_regressors


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Frequently causes travis to time out")
@pytest.mark.slow
def test_validation(network=MLPClassifier()):
    """
    testing that the networks log validation predictions in the history object
    """

    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)

    X_train = X_train[:10]
    y_train = y_train[:10]

    if isinstance(network, BaseRegressor):
        # Create some regression values, taken from test_regressor
        y_train = np.zeros(len(y_train))
        for i in range(len(X_train)):
            y_train[i] = X_train.iloc[i].iloc[0].iloc[0]

    network.fit(X_train, y_train, validation_X=X_train, validation_y=y_train)
    hist = network.history.history

    assert ('val_loss' in hist)
    assert (isinstance(hist['val_loss'][0],
                       (float, np.single, np.double, np.float32, np.float64)))


@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
                    reason="Frequently causes travis to time out")
@pytest.mark.slow
def test_all_networks():
    networks = {
        **construct_all_classifiers(SMALL_NB_EPOCHS),
        **construct_all_regressors(SMALL_NB_EPOCHS),
    }

    # these networks do not support validation data as yet
    networks.pop('MCNNClassifier_quick')
    networks.pop('TWIESNClassifier_quick')

    # networks = [
    #     MLPClassifier(nb_epochs=SMALL_NB_EPOCHS),
    #     ResNetClassifier(nb_epochs=SMALL_NB_EPOCHS),
    #     InceptionTimeClassifier(nb_epochs=SMALL_NB_EPOCHS),
    # ]

    for name, network in networks.items():
        print("\n\t\t" + name + " validation testing started")
        test_validation(network)
        print("\t\t" + name + " validation testing finished")


if __name__ == "__main__":
    test_all_networks()
