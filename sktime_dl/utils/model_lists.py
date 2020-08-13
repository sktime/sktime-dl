from sktime_dl.deeplearning import CNNClassifier
from sktime_dl.deeplearning import CNNRegressor
from sktime_dl.deeplearning import EncoderClassifier
from sktime_dl.deeplearning import EncoderRegressor
from sktime_dl.deeplearning import FCNClassifier
from sktime_dl.deeplearning import FCNRegressor
from sktime_dl.deeplearning import InceptionTimeClassifier
from sktime_dl.deeplearning import InceptionTimeRegressor
from sktime_dl.deeplearning import LSTMRegressor
from sktime_dl.deeplearning import MCDCNNClassifier
from sktime_dl.deeplearning import MCDCNNRegressor
from sktime_dl.deeplearning import MCNNClassifier
from sktime_dl.deeplearning import MLPClassifier
from sktime_dl.deeplearning import MLPRegressor
from sktime_dl.deeplearning import ResNetClassifier
from sktime_dl.deeplearning import ResNetRegressor
from sktime_dl.deeplearning import SimpleRNNRegressor
from sktime_dl.deeplearning import TLENETClassifier
from sktime_dl.deeplearning import TLENETRegressor
from sktime_dl.deeplearning import TWIESNClassifier

SMALL_NB_EPOCHS = 3


def construct_all_classifiers(nb_epochs=None):
    """
    Creates a list of all classification networks ready for testing

    :param nb_epochs: int, if not None, value shall be set for all networks
    that accept it
    :return: map of strings to sktime_dl BaseDeepRegressor implementations
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
            'ResNetRegressor': ResNetRegressor(),
            'TLENETRegressor': TLENETRegressor(),
            'InceptionTimeRegressor': InceptionTimeRegressor(),
            'SimpleRNNRegressor': SimpleRNNRegressor(),
        }
