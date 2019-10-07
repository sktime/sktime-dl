import gc
import sys

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
from sktime_dl.classifiers.deeplearning import TunedCNNClassifier

import sktime.contrib.experiments as exp

def setNetwork(cls, resampleId=0):
    """
    Basic way of determining the classifier to build. To differentiate settings just and another elif. So, for example, if
    you wanted tuned TSF, you just pass TuneTSF and set up the tuning mechanism in the elif.
    This may well get superceded, it is just how e have always done it
    :param cls: String indicating which classifier you want
    :return: A classifier.

    """
    if cls.lower() == 'dl4tsc_cnn':
        return CNNClassifier()
    elif cls.lower() == 'dl4tsc_encoder':
        return EncoderClassifier()
    elif cls.lower() == 'dl4tsc_fcn':
        return FCNClassifier()
    elif cls.lower() == 'dl4tsc_mcdcnn':
        return MCDCNNClassifier()
    elif  cls.lower() == 'dl4tsc_mcnn':
        return MCNNClassifier()
    elif cls.lower() == 'dl4tsc_mlp':
        return MLPClassifier()
    elif cls.lower() == 'dl4tsc_resnet':
        return ResNetClassifier()
    elif cls.lower() == 'dl4tsc_tlenet':
        return TLENETClassifier()
    elif cls.lower() == 'dl4tsc_twiesn':
        return TWIESNClassifier()
    elif cls.lower() == 'dl4tsc_tunedcnn':
        return TunedCNNClassifier()
    else:
        raise Exception('UNKNOWN CLASSIFIER')

def dlExperiment(data_dir, res_dir, classifier_name, dset, fold, classifier=None):

    if classifier is None:
        classifier = setNetwork(classifier_name, fold)

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
        "dl4tsc_tunedcnn"
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
        TunedCNNClassifier(),
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
    #allComparisonExperiments()
    dlExperiment(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))