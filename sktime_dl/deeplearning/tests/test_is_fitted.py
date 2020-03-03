import numpy as np

from sklearn.exceptions import NotFittedError

from sktime.datasets import load_italy_power_demand

from sktime_dl.deeplearning import *


def test_is_fitted(network=CNNClassifier(nb_epochs=5)):
    '''
    testing that the networks correctly recognise when they are not fitted
    '''

    X_train, y_train = load_italy_power_demand("TRAIN", return_X_y=True)

    if ('Regressor' in network.__class__.__name__):
        # Create some regression values, taken from test_regressor
        y_train = np.zeros(len(y_train))
        for i in range(len(X_train)):
            y_train[i] = X_train.iloc[i].iloc[0].iloc[0]

    # first try to predict without fitting: SHOULD fail
    try:
        network.predict(X_train[:10])
        raise RuntimeError("% computed predict without being fitted first." % (network))
    except NotFittedError:
        pass  # correct behaviour

    # now try predicting after fitting, should NOT fail
    network.fit(X_train[:10], y_train[:10])
    network.predict(X_train[:10])


def test_all_networks():
    networks = [
        CNNClassifier(nb_epochs=1),
        EncoderClassifier(nb_epochs=1),
        FCNClassifier(nb_epochs=1),
        MCDCNNClassifier(nb_epochs=1),
        MCNNClassifier(nb_epochs=1),
        MLPClassifier(nb_epochs=1),
        ResNetClassifier(nb_epochs=1),
        TLENETClassifier(nb_epochs=1),
        TWIESNClassifier(),
        InceptionTimeClassifier(nb_epochs=1),

        CNNRegressor(nb_epochs=1),
        EncoderRegressor(nb_epochs=1),
        FCNRegressor(nb_epochs=1),
        # MCDCNNRegressor(nb_epochs=5), # not implemented
        # MCNNRegressor(nb_epochs=5), # not implemented
        MLPRegressor(nb_epochs=1),
        ResNetRegressor(nb_epochs=1),
        TLENETRegressor(nb_epochs=1),
        # TWIESNRegressor(), # not implemented
        InceptionTimeRegressor(nb_epochs=1),
    ]

    for network in networks:
        print('\n\t\t' + network.__class__.__name__ + ' is_fitted testing started')
        test_is_fitted(network)
        print('\t\t' + network.__class__.__name__ + ' is_fitted testing finished')


if __name__ == "__main__":
    test_all_networks()
