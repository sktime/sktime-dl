# -*- coding: utf-8 -*-
"""Experiments: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format
todo: Tidy up this file!
"""

import os

import sklearn.preprocessing
import sklearn.utils



os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sys
import time
import numpy as np
import pandas as pd


from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sktime.contrib.experiments import run_experiment
from sktime.datasets.base import load_UCR_UEA_dataset
from sktime.contrib.experiments import write_results_to_uea_format


from sktime_dl.classification import CNNClassifier
from sktime_dl.classification import (
    CNNClassifier,
    CNTCClassifier,
    EncoderClassifier,
    FCNClassifier,
    InceptionTimeClassifier,
    LSTMFCNClassifier,
    MCDCNNClassifier,
    MCNNClassifier,
    MLPClassifier,
    ResNetClassifier,
    TLENETClassifier,
    TWIESNClassifier,
)

__author__ = ["Tony Bagnall"]

"""Prototype mechanism for testing classifiers on the UCR format. This mirrors the
mechanism used in Java,
https://github.com/TonyBagnall/uea-tsc/tree/master/src/main/java/experiments
but is not yet as engineered. However, if you generate results using the method
recommended here, they can be directly and automatically compared to the results
generated in java

"""

estimator_list = [
    "CNNClassifier",
    "EncoderClassifier",
    "FCNClassifier",
    "InceptionTimeClassifier",
    "MCDCNNClassifier",
    "MCNNClassifier",
    "MLPClassifier",
    "ResNetClassifier",
    "TLENETClassifier",
    "TWIESNClassifier",
]

#classifier_list = ["cnn", "encode", "fcn", "lstm", "mcdcnn", "mcnn", "mlp", "resnet", "tlenet", "twiesn"]
classifier_list = ["cnn"]



def set_classifier(cls, resampleId=None):
    """Construct a classifier.

    Basic way of creating the classifier to build using the default settings. This
    set up is to help with batch jobs for multiple problems to facilitate easy
    reproducability. You can set up bespoke classifier in many other ways.

    Parameters
    ----------
    cls: String indicating which classifier you want
    resampleId: classifier random seed

    Return
    ------
    A classifier.
    """
    name = cls.lower()
    # Convolutional
    if name == "cnn" or name == "cnnclassifier":
        return CNNClassifier(random_state=resampleId)
    elif name == "encode":
        return EncoderClassifier()
    elif name == "fcn":
        return FCNClassifier()
    elif name == "inceptiontime":
        return InceptionTimeClassifier()
    elif name == "mcdcnn":
        return MCDCNNClassifier()
    elif name == "mcnn":
        return MCNNClassifier()
    elif name == "mlp":
        return MLPClassifier()
    elif name == "resnet":
        return ResNetClassifier()
    elif name == "tlenet":
        return TLENETClassifier()
    elif name == "twiesn":
        return TWIESNClassifier()
    else:
        raise Exception("UNKNOWN CLASSIFIER")


def run_experiment(
    results_path,
    trainX, trainY,
    testX, testY,
    cls_name,
    dataset,
    classifier=None,
    resampleID=0,
    overwrite=False,
    format=".ts",
    train_file=False,
):
    """Run a classification experiment.

    Method to run a basic experiment and write the results to files called
    testFold<resampleID>.csv and, if required, trainFold<resampleID>.csv.

    Parameters
    ----------
    problem_path: Location of problem files, full path.
    results_path: Location of where to write results. Any required directories
        will be created
    cls_name: determines which classifier to use, as defined in set_classifier.
        This assumes predict_proba is
    implemented, to avoid predicting twice. May break some classifiers though
    dataset: Name of problem. Files must be  <problem_path>/<dataset>/<dataset>+
                "_TRAIN"+format, same for "_TEST"
    resampleID: Seed for resampling. If set to 0, the default train/test split
                from file is used. Also used in output file name.
    overwrite: if set to False, this will only build results if there is not a
                result file already present. If
    True, it will overwrite anything already there
    format: Valid formats are ".ts", ".arff" and ".long".
    For more info on format, see   examples/Loading%20Data%20Examples.ipynb
    train_file: whether to generate train files or not. If true, it performs a
                10xCV on the train and saves
    """
    build_test = True
    if not overwrite:
        full_path = (
            str(results_path)
            + "/"
            + str(cls_name)
            + "/Predictions/"
            + str(dataset)
            + "/testFold"
            + str(resampleID)
            + ".csv"
        )
        if os.path.exists(full_path):
            print(
                full_path
                + " Already exists and overwrite set to false, not building Test"
            )
            build_test = False
        if train_file:
            full_path = (
                str(results_path)
                + "/"
                + str(cls_name)
                + "/Predictions/"
                + str(dataset)
                + "/trainFold"
                + str(resampleID)
                + ".csv"
            )
            if os.path.exists(full_path):
                print(
                    full_path
                    + " Already exists and overwrite set to false, not building Train"
                )
                train_file = False
        if train_file == False and build_test == False:
            return

    # TO DO: Automatically differentiate between problem types,
    # currently only works with .ts
    #trainX, trainY = load_ts(problem_path + dataset + "/" + dataset + "_TRAIN" +
    # format)
    #testX, testY = load_ts(problem_path + dataset + "/" + dataset + "_TEST" + format)
    if resampleID != 0:
        # allLabels = np.concatenate((trainY, testY), axis = None)
        # allData = pd.concat([trainX, testX])
        # train_size = len(trainY) / (len(trainY) + len(testY))
        # trainX, testX, trainY, testY = train_test_split(allData, allLabels,
        # train_size=train_size,
        # random_state=resampleID, shuffle=True,
        # stratify=allLabels)
        trainX, trainY, testX, testY = stratified_resample(
            trainX, trainY, testX, testY, resampleID
        )

    le = preprocessing.LabelEncoder()
    le.fit(trainY)
    trainY = le.transform(trainY)
    testY = le.transform(testY)
    if classifier is None:
        classifier = set_classifier(cls_name, resampleID)
    print(cls_name + " on " + dataset + " resample number " + str(resampleID))
    if build_test:
        # TO DO : use sklearn CV
        start = int(round(time.time() * 1000))
        classifier.fit(trainX, trainY)
        build_time = int(round(time.time() * 1000)) - start
        start = int(round(time.time() * 1000))
        probs = classifier.predict_proba(testX)
        preds = classifier.classes_[np.argmax(probs, axis=1)]
        test_time = int(round(time.time() * 1000)) - start
        ac = accuracy_score(testY, preds)
        print(
            cls_name
            + " on "
            + dataset
            + " resample number "
            + str(resampleID)
            + " test acc: "
            + str(ac)
            + " time: "
            + str(test_time)
        )
        #        print(str(classifier.findEnsembleTrainAcc(trainX, trainY)))
        if "Composite" in cls_name:
            second = "Para info too long!"
        else:
            second = str(classifier.get_params())
        second.replace("\n", " ")
        second.replace("\r", " ")

        print(second)
        temp = np.array_repr(classifier.classes_).replace("\n", "")

        third = (
            str(ac)
            + ","
            + str(build_time)
            + ","
            + str(test_time)
            + ",-1,-1,"
            + str(len(classifier.classes_))
        )
        write_results_to_uea_format(
            second_line=second,
            third_line=third,
            output_path=results_path,
            classifier_name=cls_name,
            resample_seed=resampleID,
            predicted_class_vals=preds,
            actual_probas=probs,
            dataset_name=dataset,
            actual_class_vals=testY,
            split="TEST",
        )
    if train_file:
        start = int(round(time.time() * 1000))
        if build_test and hasattr(
            classifier, "_get_train_probs"
        ):  # Normally Can only do this if test has been built
            train_probs = classifier._get_train_probs(trainX)
        else:
            train_probs = cross_val_predict(
                classifier, X=trainX, y=trainY, cv=10, method="predict_proba"
            )
        train_time = int(round(time.time() * 1000)) - start
        train_preds = classifier.classes_[np.argmax(train_probs, axis=1)]
        train_acc = accuracy_score(trainY, train_preds)
        print(
            cls_name
            + " on "
            + dataset
            + " resample number "
            + str(resampleID)
            + " train acc: "
            + str(train_acc)
            + " time: "
            + str(train_time)
        )
        if "Composite" in cls_name:
            second = "Para info too long!"
        else:
            second = str(classifier.get_params())
        second.replace("\n", " ")
        second.replace("\r", " ")
        temp = np.array_repr(classifier.classes_).replace("\n", "")
        third = (
            str(train_acc)
            + ","
            + str(train_time)
            + ",-1,-1,-1,"
            + str(len(classifier.classes_))
        )
        write_results_to_uea_format(
            second_line=second,
            third_line=third,
            output_path=results_path,
            classifier_name=cls_name,
            resample_seed=resampleID,
            predicted_class_vals=train_preds,
            actual_probas=train_probs,
            dataset_name=dataset,
            actual_class_vals=trainY,
            split="TRAIN",
        )



if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing
    """
    print(" Local Run")
    results_dir = "C:/Temp/sktime-dl/"
    classifier = "resnet"
    resample = 0
    #         for i in range(0, len(univariate_datasets)):
    #             dataset = univariate_datasets[i]
    # #            print(i)
    # #            print(" problem = "+dataset)
    problem="ItalyPowerDemand"
    print("Loading ",problem)
    trX, trY = load_UCR_UEA_dataset(problem, split="train", return_X_y=True)
    teX, teY = load_UCR_UEA_dataset(problem, split="test", return_X_y=True)
    tf = False
    run_experiment(
        overwrite=True,
        trainX=trX,
        trainY=trY,
        testX=trX,
        testY=trY,
        results_path=results_dir,
        cls_name=classifier,
        dataset=problem,
        resampleID=resample,
        train_file=tf,
    )
