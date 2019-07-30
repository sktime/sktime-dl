import gc
import sys

import keras

from sktime.datasets import load_italy_power_demand #, load_basic_motions

from sktime_dl.classifiers.deeplearning import CNNClassifier
from sktime_dl.classifiers.deeplearning import EncoderClassifier
from sktime_dl.classifiers.deeplearning import FCNClassifier
from sktime_dl.classifiers.deeplearning import MCDCNNClassifier
from sktime_dl.classifiers.deeplearning import MCNNClassifier
from sktime_dl.classifiers.deeplearning import MLPClassifier
from sktime_dl.classifiers.deeplearning import ResNetClassifier
from sktime_dl.classifiers.deeplearning import TLENETClassifier
from sktime_dl.classifiers.deeplearning import TWIESNClassifier
from sktime_dl.classifiers.deeplearning import TunedCNNClassifier


def test_basic_univariate(network=CNNClassifier()):
    '''
    just a super basic test with gunpoint,
        load data,
        construct classifier,
        fit,
        score
    '''

    print("Start test_basic()")

    X_train, y_train = load_italy_power_demand(split='TRAIN', return_X_y=True)
    X_test, y_test = load_italy_power_demand(split='TEST', return_X_y=True)

    hist = network.fit(X_train[:10], y_train[:10])

    print(network.score(X_test[:10], y_test[:10]))
    print("End test_basic()")


def test_pipeline(network=CNNClassifier()):
    '''
    slightly more generalised test with sktime pipelines
        load data,
        construct pipeline with classifier,
        fit,
        score
    '''

    print("Start test_pipeline()")

    from sktime.pipeline import Pipeline

    # just a simple (useless) pipeline for the purposes of testing
    # that the keras network is compatible with that system
    steps = [
        ('clf', network)
    ]
    clf = Pipeline(steps)

    X_train, y_train = load_italy_power_demand(split='TRAIN', return_X_y=True)
    X_test, y_test = load_italy_power_demand(split='TEST', return_X_y=True)

    hist = clf.fit(X_train[:10], y_train[:10])

    print(clf.score(X_test[:10], y_test[:10]))
    print("End test_pipeline()")


def test_highLevelsktime(network=CNNClassifier()):
    '''
    truly generalised test with sktime tasks/strategies
        load data, build task
        construct classifier, build strategy
        fit,
        score
    '''

    print("start test_highLevelsktime()")

    from sktime.highlevel import TSCTask
    from sktime.highlevel import TSCStrategy
    from sklearn.metrics import accuracy_score

    train = load_italy_power_demand(split='TRAIN')
    test = load_italy_power_demand(split='TEST')
    task = TSCTask(target='class_val', metadata=train)

    strategy = TSCStrategy(network)
    strategy.fit(task, train.iloc[:10])

    y_pred = strategy.predict(test.iloc[:10])
    y_test = test.iloc[:10][task.target]
    print(accuracy_score(y_test, y_pred))

    print("End test_highLevelsktime()")


# def test_basic_multivariate(network=cnn.CNN()):
#     '''
#     just a super basic test with basicmotions,
#         load data,
#         construct classifier,
#         fit,
#         score
#     '''
#     print("Start test_multivariate()")
#
#     X_train, y_train = load_basic_motions(split='TRAIN', return_X_y=True)
#     X_test, y_test = load_basic_motions(split='TRAIN', return_X_y=True)
#
#     hist = network.fit(X_train, y_train)
#
#     print(network.score(X_test, y_test))
#     print("End test_multivariate()")


def test_network(network=CNNClassifier()):
    # sklearn compatibility
    # check_estimator(FCN)

    test_basic_univariate(network)
    # test_basic_multivariate(network)
    test_pipeline(network)
    test_highLevelsktime(network)


def all_networks_all_tests():

    networks = [
        CNNClassifier(),
        EncoderClassifier(),
        FCNClassifier(),
        MCDCNNClassifier(),
        MCNNClassifier(),
        MLPClassifier(),
        ResNetClassifier(),
        TLENETClassifier(),
        TWIESNClassifier(),
        TunedCNNClassifier(),
    ]

    for network in networks:
        print('\n\t\t' + network.__class__.__name__ + ' testing started')
        test_network(network)
        print('\t\t' + network.__class__.__name__ + ' testing finished')


def comparisonExperiments():
    data_dir = sys.argv[1]
    res_dir = sys.argv[2]

    complete_classifiers = [
        "dl4tsc_cnn",
        "dl4tsc_encoder",
        "dl4tsc_fcn",
        "dl4tsc_mcdcnn",
        "dl4tsc_mcnn",
        "dl4tsc_mlp",
        "dl4tsc_resnet",
        "dl4tsc_tlenet",
        "dl4tsc_twiesn",
    ]

    small_datasets = [
        "Beef",
        "Car",
        "Coffee",
        "CricketX",
        "CricketY",
        "CricketZ",
        "DiatomSizeReduction",
        "Fish",
        "GunPoint",
        "ItalyPowerDemand",
        "MoteStrain",
        "OliveOil",
        "Plane",
        "SonyAIBORobotSurface1",
        "SonyAIBORobotSurface2",
        "SyntheticControl",
        "Trace",
        "TwoLeadECG",
    ]

    num_folds = 30

    import sktime.contrib.experiments as exp

    for f in range(num_folds):
        for d in small_datasets:
            for c in complete_classifiers:
                print(c, d, f)
                try:
                    exp.run_experiment(data_dir, res_dir, c, d, f)
                    gc.collect()
                    keras.backend.clear_session()
                except:
                    print('\n\n FAILED: ', sys.exc_info()[0], '\n\n')


if __name__ == "__main__":
    all_networks_all_tests()
