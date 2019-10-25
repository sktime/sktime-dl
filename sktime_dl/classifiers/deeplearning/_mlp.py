# Multi layer perceptron, adapted from the implementation from Fawaz et. al
# https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py
#
# Network originally proposed by:
#
# @inproceedings{wang2017time,
#   title={Time series classification from scratch with deep neural networks: A strong baseline},
#   author={Wang, Zhiguang and Yan, Weizhong and Oates, Tim},
#   booktitle={2017 International joint conference on neural networks (IJCNN)},
#   pages={1578--1585},
#   year={2017},
#   organization={IEEE}
# }

__author__ = "James Large"

import keras
import numpy as np
import pandas as pd

from sktime_dl.classifiers.deeplearning._base import BaseDeepClassifier


class MLPClassifier(BaseDeepClassifier):

    def __init__(self,
                 nb_epochs = 5000,
                 batch_size=16,

                 random_seed=0,
                 verbose=False,
                 model_save_directory=None):

        self.verbose = verbose
        self.model_save_directory = model_save_directory

        # calced in fit
        self.classes_ = None
        self.nb_classes = -1
        self.input_shape = None
        self.model = None
        self.history = None

        # predefined
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.callbacks = None

        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)

    def build_model(self, input_shape, nb_classes, **kwargs):
        input_layer = keras.layers.Input(input_shape)

        # flatten/reshape because when multivariate all should be on the same axis
        input_layer_flattened = keras.layers.Flatten()(input_layer)

        layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
        layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

        layer_2 = keras.layers.Dropout(0.2)(layer_1)
        layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

        layer_3 = keras.layers.Dropout(0.2)(layer_2)
        layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)

        output_layer = keras.layers.Dropout(0.3)(layer_3)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

        self.callbacks = [reduce_lr]

        return model

    def fit(self, X, y, **kwargs):

        X = self.check_and_clean_data(X)

        y_onehot = self.convert_y(y)
        self.input_shape = X.shape[1:]

        self.batch_size = int(min(X.shape[0] / 10, self.batch_size))

        self.model = self.build_model(self.input_shape, self.nb_classes)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(X, y_onehot, batch_size=self.batch_size, epochs=self.nb_epochs,
                                      verbose=self.verbose, callbacks=self.callbacks)

        self.save_trained_model()