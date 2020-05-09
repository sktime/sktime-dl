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
    :return: list of sktime_dl BaseDeepClassifier imeplementations
    """
    if nb_epochs is not None:
        # potentially quicker versions for tests
        return [
            CNNClassifier(nb_epochs=nb_epochs),
            EncoderClassifier(nb_epochs=nb_epochs),
            FCNClassifier(nb_epochs=nb_epochs),
            MCDCNNClassifier(nb_epochs=nb_epochs),
            MCNNClassifier(nb_epochs=nb_epochs),
            MLPClassifier(nb_epochs=nb_epochs),
            ResNetClassifier(nb_epochs=nb_epochs),
            TLENETClassifier(nb_epochs=nb_epochs),
            TWIESNClassifier(),
            InceptionTimeClassifier(nb_epochs=nb_epochs),
        ]
    else:
        # the 'literature-conforming' versions
        return [
            CNNClassifier(),
            EncoderClassifier(),
            FCNClassifier(),
            MCDCNNClassifier(),
            MCNNClassifier(),
            MLPClassifier(),
            ResNetClassifier(),
            TLENETClassifier(),
            TWIESNClassifier(),
            InceptionTimeClassifier(),
        ]


def construct_all_regressors(nb_epochs=None):
    """
    Creates a list of all regression networks ready for testing

    :param nb_epochs: int, if not None, value shall be set for all networks
    that accept it
    :return: list of sktime_dl BaseDeepRegressor imeplementations
    """
    if nb_epochs is not None:
        # potentially quicker versions for tests
        return [
            CNNRegressor(nb_epochs=nb_epochs, kernel_size=3, avg_pool_size=1),
            EncoderRegressor(nb_epochs=nb_epochs),
            FCNRegressor(nb_epochs=nb_epochs),
            LSTMRegressor(nb_epochs=nb_epochs),
            MCDCNNRegressor(nb_epochs=nb_epochs, dense_units=1),
            MLPRegressor(nb_epochs=nb_epochs),
            ResNetRegressor(nb_epochs=nb_epochs),
            TLENETRegressor(nb_epochs=nb_epochs),
            InceptionTimeRegressor(nb_epochs=nb_epochs),
            SimpleRNNRegressor(nb_epochs=nb_epochs),
        ]
    else:
        # the 'literature-conforming' versions
        return [
            CNNRegressor(),
            EncoderRegressor(),
            FCNRegressor(),
            LSTMRegressor(),
            MCDCNNRegressor(),
            MLPRegressor(),
            ResNetRegressor(),
            TLENETRegressor(),
            InceptionTimeRegressor(),
            SimpleRNNRegressor(),
        ]
