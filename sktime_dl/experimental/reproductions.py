import sys

if len(sys.argv) > 1:
    setseed = bool(sys.argv[6])
    if setseed:
        # rngseed = int(sys.argv[5])
        rngseed = int(sys.argv[7]) + 5
        from numpy.random import seed

        seed(rngseed)
        from tensorflow import set_random_seed

        set_random_seed(rngseed)

import gc
import keras
from sktime.contrib.experiments import univariate_datasets

from sktime_dl.classifiers.deeplearning import CNNClassifier
from sktime_dl.classifiers.deeplearning import EncoderClassifier
from sktime_dl.classifiers.deeplearning import FCNClassifier
from sktime_dl.classifiers.deeplearning import MCDCNNClassifier
from sktime_dl.classifiers.deeplearning import MCNNClassifier
from sktime_dl.classifiers.deeplearning import MLPClassifier
from sktime_dl.classifiers.deeplearning import ResNetClassifier
from sktime_dl.classifiers.deeplearning import TLENETClassifier
from sktime_dl.classifiers.deeplearning import TWIESNClassifier
from sktime_dl.classifiers.deeplearning import InceptionTimeClassifier
from sktime_dl.classifiers.deeplearning import DeepLearnerEnsembleClassifier

import sktime.contrib.experiments as exp


def setNetwork(data_dir, res_dir, cls, dset, fold, classifier=None):
    """
    Basic way of determining the classifier to build. To differentiate settings just and another elif. So, for example, if
    you wanted tuned TSF, you just pass TuneTSF and set up the tuning mechanism in the elif.
    This may well get superceded, it is just how e have always done it
    :param cls: String indicating which classifier you want
    :return: A classifier.

    """
    fold = int(fold)
    if cls.lower() == 'dl4tsc_cnn':
        return CNNClassifier(random_seed=fold)
    elif cls.lower() == 'dl4tsc_encoder':
        return EncoderClassifier(random_seed=fold)
    elif cls.lower() == 'dl4tsc_fcn':
        return FCNClassifier(random_seed=fold)
    elif cls.lower() == 'dl4tsc_mcdcnn':
        return MCDCNNClassifier(random_seed=fold)
    elif cls.lower() == 'dl4tsc_mcnn':
        return MCNNClassifier(random_seed=fold)
    elif cls.lower() == 'dl4tsc_mlp':
        return MLPClassifier(random_seed=fold)
    elif cls.lower() == 'dl4tsc_resnet':
        return ResNetClassifier(random_seed=fold)
    elif cls.lower() == 'dl4tsc_tlenet':
        return TLENETClassifier(random_seed=fold)
    elif cls.lower() == 'dl4tsc_twiesn':
        return TWIESNClassifier(random_seed=fold)
    elif cls.lower() == 'dl4tsc_tunedcnn':
        return TunedCNNClassifier(random_seed=fold)
    elif cls.lower() == "inception0":
        return InceptionTimeClassifier(random_seed=fold)
    elif cls.lower() == "inception1":
        return InceptionTimeClassifier(random_seed=fold)
    elif cls.lower() == "inception2":
        return InceptionTimeClassifier(random_seed=fold)
    elif cls.lower() == "inception3":
        return InceptionTimeClassifier(random_seed=fold)
    elif cls.lower() == "inception4":
        return InceptionTimeClassifier(random_seed=fold)
    elif cls.lower() == "inceptiontime":
        return DeepLearnerEnsembleClassifier(res_dir, dset, random_seed=fold, network_name='inception', nb_iterations=5)
    else:
        raise Exception('UNKNOWN CLASSIFIER: ' + cls)


def dlExperiment(data_dir, res_dir, classifier_name, dset, fold, classifier=None):
    if classifier is None:
        classifier = setNetwork(data_dir, res_dir, classifier_name, dset, fold, classifier=None)

    exp.run_experiment(data_dir, res_dir, classifier_name, dset, classifier=classifier, resampleID=fold)


def allComparisonExperiments():
    data_dir = sys.argv[1]
    res_dir = sys.argv[2]

    classifier_names = [
        "dl4tsc_cnn",
        "dl4tsc_encoder",
        "dl4tsc_fcn",
        "dl4tsc_mcdcnn",
        "dl4tsc_mcnn",
        "dl4tsc_mlp",
        "dl4tsc_resnet",
        "dl4tsc_tlenet",
        "dl4tsc_twiesn",
        "dl4tsc_tunedcnn",
        "inception_single",
        "inception_time"
    ]

    classifiers = [
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
        DeepLearnerEnsembleClassifier(network_name="InceptionTimeClassifier")
    ]

    num_folds = 30

    for f in range(num_folds):
        for d in univariate_datasets:
            for cname, c in zip(classifier_names, classifiers):
                print(cname, d, f)
                try:
                    dlExperiment(data_dir, res_dir, cname, d, f, classifier=c)
                    gc.collect()
                    keras.backend.clear_session()
                except:
                    print('\n\n FAILED: ', sys.exc_info()[0], '\n\n')


if __name__ == "__main__":
    # allComparisonExperiments()

    classifier = sys.argv[3]
    if classifier == "inception":  # seeding inception ensemble exps for bakeoff redux
       classifier = classifier + sys.argv[7]

    dlExperiment(sys.argv[1], sys.argv[2], classifier, sys.argv[4], int(sys.argv[5]))

