"""

To run a dl experiment:

    sys.argv[1] : string, data read directory
    sys.argv[2] : string, results write directory
    sys.argv[3] : string, classifier name (from list in setNetwork) todo make software engineer-y
    sys.argv[1] : string, dataset same
    sys.argv[1] : int, resample id/high-level experimental seed
    sys.argv[1] : bool, whether to set numpy/tf seed for model initialisations
    sys.argv[1] : int, numpy/tf seed for model initialisations

    dlExperiment(sys.argv[1], sys.argv[2], classifier, sys.argv[4], int(sys.argv[5]))

"""

import sys
import os

if len(sys.argv) > 6:
    setseed = bool(sys.argv[6])
    if setseed:
        # rngseed = int(sys.argv[5])
        rngseed = int(sys.argv[7]) + 5
        from numpy.random import seed

        seed(rngseed)

        # from tensorflow import set_random_state
        # set_random_state(rngseed)

        import tensorflow as tf

        tf.random.set_seed(rngseed)

import gc
from tensorflow import keras

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
from sktime_dl.meta import EnsembleFromFileClassifier

#import sktime.contrib.experiments as exp
import sktime_dl.experimental.dlexp as dlexp

ucr112dsets = [
    "ACSF1",
    "Adiac",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
    "Car",
    "CBF",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "HouseTwenty",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
]

ueamv26dsets = [
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "EthanolConcentration",
    "ERing",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PenDigits",
    "PEMS-SF",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "StandWalkJump",
    "UWaveGestureLibrary",
]

def setNetwork(data_dir, res_dir, cls, dset, fold, classifier=None):
    """
    Basic way of determining the classifier to build. To differentiate settings just and another elif. So, for example, if
    you wanted tuned TSF, you just pass TuneTSF and set up the tuning mechanism in the elif.
    This may well get superceded, it is just how e have always done it
    :param cls: String indicating which classifier you want
    :return: A classifier.

    """

    model_save_dir = res_dir + cls + "/Models/" + dset + "/"
    model_name = cls + "_" + dset + "_" + str(fold)

    try:
        os.makedirs(model_save_dir)
    except os.error:
        pass  # raises os.error if path already exists

    fold = int(fold)
    if cls.lower() == "cnn":
        return CNNClassifier(random_state=fold, model_name=model_name, model_save_directory=model_save_dir)
    elif cls.lower() == "encoder":
        return EncoderClassifier(random_state=fold, model_name=model_name, model_save_directory=model_save_dir)
    elif cls.lower() == "fcn":
        return FCNClassifier(random_state=fold, model_name=model_name, model_save_directory=model_save_dir)
    elif cls.lower() == "mcdcnn":
        return MCDCNNClassifier(random_state=fold, model_name=model_name, model_save_directory=model_save_dir)
    elif cls.lower() == "mcnn":
        return MCNNClassifier(random_state=fold, model_name=model_name, model_save_directory=model_save_dir)
    elif cls.lower() == "mlp":
        return MLPClassifier(random_state=fold, model_name=model_name, model_save_directory=model_save_dir)
    elif cls.lower() == "resnet":
        return ResNetClassifier(random_state=fold, model_name=model_name, model_save_directory=model_save_dir)
    elif cls.lower() == "tlenet":
        return TLENETClassifier(random_state=fold, model_name=model_name, model_save_directory=model_save_dir)
    elif cls.lower() == "twiesn":
        return TWIESNClassifier(random_state=fold, model_name=model_name, model_save_directory=model_save_dir)
    elif cls.lower() == "inception0":
        return InceptionTimeClassifier(random_state=fold, model_name=model_name, model_save_directory=model_save_dir)
    elif cls.lower() == "inception1":
        return InceptionTimeClassifier(random_state=fold, model_name=model_name, model_save_directory=model_save_dir)
    elif cls.lower() == "inception2":
        return InceptionTimeClassifier(random_state=fold, model_name=model_name, model_save_directory=model_save_dir)
    elif cls.lower() == "inception3":
        return InceptionTimeClassifier(random_state=fold, model_name=model_name, model_save_directory=model_save_dir)
    elif cls.lower() == "inception4":
        return InceptionTimeClassifier(random_state=fold)
    elif cls.lower() == "inceptiontime":
        return EnsembleFromFileClassifier(
            res_dir,
            dset,
            random_state=fold,
            network_name="inception",
            nb_iterations=5,
        )
    else:
        raise Exception("UNKNOWN CLASSIFIER: " + cls)


def dlExperiment(
        data_dir, res_dir, classifier_name, dset, fold, classifier=None
):
    if classifier is None:
        classifier = setNetwork(
            data_dir, res_dir, classifier_name, dset, fold, classifier=None
        )

    dlexp.run_experiment(
        data_dir,
        res_dir,
        classifier_name,
        dset,
        classifier=classifier,
        resampleID=fold,
    )


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
        "inception0",
        "inception1",
        "inception2",
        "inception3",
        "inception4",
        "inceptiontime",
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
    ]

    num_folds = 30

    for f in range(num_folds):
        for d in ucr112dsets:
            for cname, c in zip(classifier_names, classifiers):
                print(cname, d, f)
                try:
                    dlExperiment(data_dir, res_dir, cname, d, f, classifier=c)
                    gc.collect()
                    keras.backend.clear_session()
                except:
                    print("\n\n FAILED: ", sys.exc_info()[0], "\n\n")


def ensembleInception(data_dir, res_dir, classifier_name, dsets, folds, overwrite=False):
    missingdsets = []

    for dset in dsets:
        for fold in folds:
            try:
                classifier = setNetwork(
                    data_dir, res_dir, classifier_name, dset, fold, classifier=None
                )
                dlexp.run_experiment(
                    data_dir,
                    res_dir,
                    classifier_name,
                    dset,
                    classifier=classifier,
                    resampleID=fold,
                    overwrite=overwrite,
                )
            except FileNotFoundError:
                missingdsets.append(dset+str(fold))
                print(dset+str(fold), " missing")

    print("\n\n\n", missingdsets)


if __name__ == "__main__":
    # allComparisonExperiments()

    classifier = sys.argv[3]
    if (
            classifier == "inception"
    ):  # seeding inception ensemble exps for bakeoff redux
        classifier = classifier + sys.argv[7]

    dlExperiment(
        sys.argv[1], sys.argv[2], classifier, sys.argv[4], int(sys.argv[5])
    )

    # ensembleInception("Z:/ArchiveData/Multivariate_ts/", "E:/MultivariateArchive/", "inceptiontime", ueamv26dsets, range(0,30), overwrite=False)
