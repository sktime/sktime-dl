'''
Compare accuracy of the classifiers against results published at
https://github.com/hfawaz/dl-4-tsc/blob/master/README.md
Test that accuracy is higher than the published accuracy minus three
standard deviations (also published) on the ItalyPowerDemand dataset.
'''

import pytest
from flaky import flaky

from sktime.datasets import load_italy_power_demand

from sktime_dl.deeplearning import CNNClassifier
from sktime_dl.deeplearning import EncoderClassifier
from sktime_dl.deeplearning import FCNClassifier
from sktime_dl.deeplearning import MCDCNNClassifier
from sktime_dl.deeplearning import MCNNClassifier
from sktime_dl.deeplearning import MLPClassifier
from sktime_dl.deeplearning import ResNetClassifier
from sktime_dl.deeplearning import TLENETClassifier
from sktime_dl.deeplearning import TWIESNClassifier
from sktime_dl.deeplearning import InceptionTimeClassifier


def is_not_value_error(err, *args):
    return not issubclass(err[0], ValueError)


ACCURACY_DEVIATION_THRESHOLD = 3  # times the std deviation of reported results, if available


def accuracy_test(network=CNNClassifier(), lower=0.94, upper=1.0):
    '''
    Test the classifier accuracy against expected lower and upper bounds.
    '''
    print("Start accuracy_test:", network.__class__.__name__)

    X_train, y_train = load_italy_power_demand(split='TRAIN', return_X_y=True)
    X_test, y_test = load_italy_power_demand(split='TEST', return_X_y=True)

    __ = network.fit(X_train, y_train)

    accuracy = network.score(X_test, y_test)
    print(network.__class__.__name__, "accuracy:", accuracy)
    assert (accuracy > lower and accuracy <= upper)


@pytest.mark.slow
@flaky(max_runs=3, rerun_filter=is_not_value_error)
def test_cnn_accuracy():
    accuracy_test(network=CNNClassifier(), lower=0.955 - ACCURACY_DEVIATION_THRESHOLD * 0.004)


@pytest.mark.slow
@flaky(max_runs=3, rerun_filter=is_not_value_error)
def test_encoder_accuracy():
    accuracy_test(network=EncoderClassifier(), lower=0.965 - ACCURACY_DEVIATION_THRESHOLD * 0.005)


@pytest.mark.slow
@flaky(max_runs=3, rerun_filter=is_not_value_error)
def test_fcn_accuracy():
    accuracy_test(network=FCNClassifier(), lower=0.961 - ACCURACY_DEVIATION_THRESHOLD * 0.003)


@pytest.mark.slow
@flaky(max_runs=3, rerun_filter=is_not_value_error)
def test_mcdcnn_accuracy():
    accuracy_test(network=MCDCNNClassifier(), lower=0.955 - ACCURACY_DEVIATION_THRESHOLD * 0.019)


@pytest.mark.skip(reason="Very slow running, causes Travis to time out.")
@pytest.mark.slow
@flaky(max_runs=3, rerun_filter=is_not_value_error)
def test_mcnn_accuracy():
    # Low accuracy is consistent with published results
    # https://github.com/hfawaz/dl-4-tsc/blob/master/README.md
    accuracy_test(network=MCNNClassifier(), lower=0.5 - ACCURACY_DEVIATION_THRESHOLD * 0.002,
                  upper=0.5 + ACCURACY_DEVIATION_THRESHOLD * 0.002)


@pytest.mark.slow
@flaky(max_runs=3, rerun_filter=is_not_value_error)
def test_mlp_accuracy():
    accuracy_test(network=MLPClassifier(), lower=0.954 - ACCURACY_DEVIATION_THRESHOLD * 0.002)


@pytest.mark.slow
@flaky(max_runs=3, rerun_filter=is_not_value_error)
def test_resnet_accuracy():
    accuracy_test(network=ResNetClassifier(), lower=0.963 - ACCURACY_DEVIATION_THRESHOLD * 0.004)


@pytest.mark.slow
@flaky(max_runs=3, rerun_filter=is_not_value_error)
def test_tlenet_accuracy():
    # Accuracy is higher than the 0.490 in the published results
    # https://github.com/hfawaz/dl-4-tsc/blob/master/README.md
    accuracy_test(network=TLENETClassifier(), lower=0.90)


@pytest.mark.slow
@flaky(max_runs=3, rerun_filter=is_not_value_error)
def test_twiesn_accuracy():
    accuracy_test(network=TWIESNClassifier(), lower=0.88 - ACCURACY_DEVIATION_THRESHOLD * 0.022)


@pytest.mark.slow
@flaky(max_runs=3, rerun_filter=is_not_value_error)
def test_inception_accuracy():
    accuracy_test(network=InceptionTimeClassifier(), lower=0.96)


if __name__ == "__main__":
    test_cnn_accuracy()
