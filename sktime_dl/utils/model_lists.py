from sktime_dl.classification import (CNNClassifier,
                                      EncoderClassifier,
                                      FCNClassifier,
                                      InceptionTimeClassifier,
                                      MCDCNNClassifier,
                                      OSCNNClassifier,
                                      TLENETClassifier,
                                      TWIESNClassifier,
                                      MCNNClassifier,
                                      MLPClassifier,
                                      ResNetClassifier
                                      )

from sktime_dl.regression import (CNNRegressor,
                                  EncoderRegressor,
                                  FCNRegressor,
                                  InceptionTimeRegressor,
                                  LSTMRegressor,
                                  MCDCNNRegressor,
                                  MLPRegressor,
                                  OSCNNRegressor,
                                  ResNetRegressor,
                                  SimpleRNNRegressor,
                                  TLENETRegressor
                                  )

SMALL_NB_EPOCHS = 3


def construct_all_classifiers(nb_epochs=None):
    """
    Creates a list of all classification networks ready for testing

    Parameters
    ----------
    nb_epochs: int, if not None, value shall be set for all networks that accept it

    Returns
    -------
    map of strings to sktime_dl BaseDeepRegressor implementations
    """
    if nb_epochs is not None:
        # potentially quicker versions for tests
        return {
            'CNNClassifier_quick': CNNClassifier(nb_epochs=nb_epochs),
            'EncoderClassifier_quick': EncoderClassifier(nb_epochs=nb_epochs),
            'FCNClassifier_quick': FCNClassifier(nb_epochs=nb_epochs),
            'MCDCNNClassifier_quick': MCDCNNClassifier(nb_epochs=nb_epochs),
            'MCNNClassifier_quick': MCNNClassifier(nb_epochs=nb_epochs),
            'MLPClassifier_quick': MLPClassifier(nb_epochs=nb_epochs),
            'OSCNNClassifier_quick': OSCNNClassifier(),
            'ResNetClassifier_quick': ResNetClassifier(nb_epochs=nb_epochs),
            'TLENETClassifier_quick': TLENETClassifier(nb_epochs=nb_epochs),
            'TWIESNClassifier_quick': TWIESNClassifier(),
            'InceptionTimeClassifier_quick': InceptionTimeClassifier(
                nb_epochs=nb_epochs),
        }
    else:
        # the 'literature-conforming' versions
        return {
            'CNNClassifier': CNNClassifier(),
            'EncoderClassifier': EncoderClassifier(),
            'FCNClassifier': FCNClassifier(),
            'MCDCNNClassifier': MCDCNNClassifier(),
            'MCNNClassifier': MCNNClassifier(),
            'MLPClassifier': MLPClassifier(),
            'OSCNNClassifier': OSCNNClassifier(),
            'ResNetClassifier': ResNetClassifier(),
            'TLENETClassifier': TLENETClassifier(),
            'TWIESNClassifier': TWIESNClassifier(),
            'InceptionTimeClassifier': InceptionTimeClassifier(),
        }


def construct_all_regressors(nb_epochs=None):
    """
    Creates a list of all regression networks ready for testing

    :param nb_epochs: int, if not None, value shall be set for all networks
    that accept it
    :return: map of strings to sktime_dl BaseDeepRegressor implementations
    """
    if nb_epochs is not None:
        # potentially quicker versions for tests
        return {
            'CNNRegressor_quick': CNNRegressor(nb_epochs=nb_epochs,
                                               kernel_size=3,
                                               avg_pool_size=1),
            'EncoderRegressor_quick': EncoderRegressor(nb_epochs=nb_epochs),
            'FCNRegressor_quick': FCNRegressor(nb_epochs=nb_epochs),
            'LSTMRegressor_quick': LSTMRegressor(nb_epochs=nb_epochs),
            'MLPRegressor_quick': MLPRegressor(nb_epochs=nb_epochs),
            'MCDCNNRegressor_quick': MCDCNNRegressor(nb_epochs=nb_epochs,
                                                     dense_units=1),
            'OSCNNRegressor_quick': OSCNNRegressor(),
            'ResNetRegressor_quick': ResNetRegressor(nb_epochs=nb_epochs),
            'TLENETRegressor_quick': TLENETRegressor(nb_epochs=nb_epochs),
            'InceptionTimeRegressor_quick': InceptionTimeRegressor(
                nb_epochs=nb_epochs),
            'SimpleRNNRegressor_quick': SimpleRNNRegressor(
                nb_epochs=nb_epochs),
        }
    else:
        # the 'literature-conforming' versions
        return {
            'CNNRegressor': CNNRegressor(),
            'EncoderRegressor': EncoderRegressor(),
            'FCNRegressor': FCNRegressor(),
            'LSTMRegressor': LSTMRegressor(),
            'MCDCNNRegressor': MCDCNNRegressor(),
            'MLPRegressor': MLPRegressor(),
            'OSCNNRegressor': OSCNNRegressor(),
            'ResNetRegressor': ResNetRegressor(),
            'TLENETRegressor': TLENETRegressor(),
            'InceptionTimeRegressor': InceptionTimeRegressor(),
            'SimpleRNNRegressor': SimpleRNNRegressor(),
        }
