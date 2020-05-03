__author__ = "James Large"

import numpy as np
from tensorflow import keras

from sktime_dl.deeplearning.base.estimators import BaseDeepClassifier
from sktime_dl.deeplearning.mlp._base import MLPNetwork
from sktime_dl.utils import check_and_clean_data


class MLPClassifier(BaseDeepClassifier, MLPNetwork):
    """Multi Layer Perceptron (MLP).

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py

    Network originally defined in:

    @inproceedings{wang2017time,
      title={Time series classification from scratch with deep neural networks: A strong baseline},
      author={Wang, Zhiguang and Yan, Weizhong and Oates, Tim},
      booktitle={2017 International joint conference on neural networks (IJCNN)},
      pages={1578--1585},
      year={2017},
      organization={IEEE}
    }
    """

    def __init__(self,
                 nb_epochs=5000,
                 batch_size=16,

                 callbacks=None,
                 random_seed=0,
                 verbose=False,
                 model_name="mlp",
                 model_save_directory=None):
        '''
        :param nb_epochs: int, the number of epochs to train the model
        :param batch_size: int, specifying the length of the 1D convolution window
        :param callbacks: list of tf.keras.callbacks.Callback objects
        :param random_seed: int, seed to any needed random actions
        :param verbose: boolean, whether to output extra information
        :param model_name: string, the name of this model for printing and file writing purposes
        :param model_save_directory: string, if not None; location to save the trained keras model in hdf5 format
        '''
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.random_seed = random_seed
        self.verbose = verbose
        self.model_name = model_name
        self.model_save_directory = model_save_directory

        self.is_fitted = False

    def build_model(self, input_shape, nb_classes, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for training
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        nb_classes: int
            The number of classes, which shall become the size of the output layer
        Returns
        -------
        output : a compiled Keras Model
        """
        input_layer, output_layer = self.build_network(input_shape, **kwargs)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        # if user hasn't provided a custom ReduceLROnPlateau via init already, add the default from literature
        if self.callbacks is None:
            self.callbacks = []

        if not any(isinstance(callback, keras.callbacks.ReduceLROnPlateau) for callback in self.callbacks):
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                          min_lr=0.0001)
            self.callbacks.append(reduce_lr)

        return model

    def fit(self, X, y, input_checks=True, **kwargs):
        """
        Build the classifier on the training set (X, y)
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame is passed, column 0 is extracted.
        y : array-like, shape = [n_instances]
            The class labels.
        input_checks: boolean
            whether to check the X and y parameters
        Returns
        -------
        self : object
        """
        if self.random_state is None:
            self.random_state = np.random.RandomState(self.random_seed)

        X = check_and_clean_data(X, y, input_checks=input_checks)
        y_onehot = self.convert_y(y)

        # ignore the number of instances, X.shape[0], just want the shape of each instance
        self.input_shape = X.shape[1:]

        self.batch_size = int(min(X.shape[0] / 10, self.batch_size))

        self.model = self.build_model(self.input_shape, self.nb_classes)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(X, y_onehot, batch_size=self.batch_size, epochs=self.nb_epochs,
                                      verbose=self.verbose, callbacks=self.callbacks)

        self.save_trained_model()
        self.is_fitted = True

        return self
