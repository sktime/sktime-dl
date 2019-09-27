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

def comparisonExperiments():
    data_dir = sys.argv[1]
    res_dir = sys.argv[2]

    complete_classifiers = [
        # "dl4tsc_cnn",
        # "dl4tsc_encoder",
        # "dl4tsc_fcn",
        # "dl4tsc_mcdcnn",
        # "dl4tsc_mcnn",
        # "dl4tsc_mlp",
        "dl4tsc_resnet",
        # "dl4tsc_tlenet",
        # "dl4tsc_twiesn",
    ]

    # small_datasets = [
    #     "Beef",
    #     "Car",
    #     "Coffee",
    #     "CricketX",
    #     "CricketY",
    #     "CricketZ",
    #     "DiatomSizeReduction",
    #     "Fish",
    #     "GunPoint",
    #     "ItalyPowerDemand",
    #     "MoteStrain",
    #     "OliveOil",
    #     "Plane",
    #     "SonyAIBORobotSurface1",
    #     "SonyAIBORobotSurface2",
    #     "SyntheticControl",
    #     "Trace",
    #     "TwoLeadECG",
    # ]

    num_folds = 30

    import sktime.contrib.experiments as exp

    for f in range(num_folds):
        for d in univariate_datasets:
            for c in complete_classifiers:
                print(c, d, f)
                try:
                    exp.run_experiment(data_dir, res_dir, c, d, f)
                    gc.collect()
                    keras.backend.clear_session()
                except:
                    print('\n\n FAILED: ', sys.exc_info()[0], '\n\n')


if __name__ == "__main__":
    comparisonExperiments()