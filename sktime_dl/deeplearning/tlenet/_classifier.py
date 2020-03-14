__author__ = "Aaron Bostrom, James Large"

from tensorflow import keras
import numpy as np

from sktime_dl.deeplearning.base.estimators import BaseDeepClassifier
from sktime_dl.deeplearning.tlenet._base import TLENETNetwork
from sktime_dl.utils import check_and_clean_data, check_is_fitted


class TLENETClassifier(BaseDeepClassifier, TLENETNetwork):
    """Time Le-Net (TLENET).

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/tlenet.py

    Network originally defined in:

    @inproceedings{le2016data,
      title={Data augmentation for time series classification using convolutional neural networks},
      author={Le Guennec, Arthur and Malinowski, Simon and Tavenard, Romain},
      booktitle={ECML/PKDD workshop on advanced analytics and learning on temporal data},
      year={2016}
    }
    """

    def __init__(self,
                 nb_epochs=1000,
                 batch_size=256,

                 callbacks=None,
                 verbose=False,
                 random_seed=0,
                 model_name="tlenet",
                 model_save_directory=None):
        super().__init__(
            model_name=model_name,
            model_save_directory=model_save_directory)
        TLENETNetwork.__init__(self, random_seed=random_seed)
        '''
        :param nb_epochs: int, the number of epochs to train the model
        :param batch_size: int, specifying the length of the 1D convolution window
        :param callbacks: list of tf.keras.callbacks.Callback objects
        :param random_seed: int, seed to any needed random actions
        :param verbose: boolean, whether to output extra information
        :param model_name: string, the name of this model for printing and file writing purposes
        :param model_save_directory: string, if not None; location to save the trained keras model in hdf5 format
        '''

        self.verbose = verbose
        self.is_fitted = False

        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.callbacks = callbacks if callbacks is not None else []

        # calced in fit
        self.input_shape = None
        self.history = None

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

        model.compile(optimizer=keras.optimizers.Adam(lr=0.01, decay=0.005),
                      loss='categorical_crossentropy', metrics=['accuracy'])

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
        X = check_and_clean_data(X, y, input_checks=input_checks)
        y_onehot = self.convert_y(y)

        # ignore the number of instances, X.shape[0], just want the shape of each instance
        self.input_shape = X.shape[1:]

        self.nb_classes = y_onehot.shape[1]

        self.adjust_parameters(X)
        X, y_onehot, tot_increase_num = self.pre_processing(X, y_onehot)

        input_shape = X.shape[1:]  # pylint: disable=E1136  # pylint/issues/3139
        self.model = self.build_model(input_shape, self.nb_classes)

        self.hist = self.model.fit(X, y_onehot, batch_size=self.batch_size, epochs=self.nb_epochs,
                                   verbose=self.verbose, callbacks=self.callbacks)

        self.save_trained_model()
        self.is_fitted = True

        return self

    def predict_proba(self, X, input_checks=True, **kwargs):
        """
        Find probability estimates for each class for all cases in X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.
            If a Pandas data frame is passed (sktime format)
            If a Pandas data frame is passed, a check is performed that it only has one column.
            If not, an exception is thrown, since this classifier does not yet have
            multivariate capability.
        input_checks: boolean
            whether to check the X parameter
        Returns
        -------
        output : array of shape = [n_instances, n_classes] of probabilities
        """
        check_is_fitted(self)

        X = check_and_clean_data(X, input_checks=input_checks)

        X, _, tot_increase_num = self.pre_processing(X)

        preds = self.model.predict(X, batch_size=self.batch_size)

        y_predicted = []
        test_num_batch = int(X.shape[0] / tot_increase_num)  # pylint: disable=E1136  # pylint/issues/3139

        ##TODO: could fix this to be an array literal.
        for i in range(test_num_batch):
            y_predicted.append(np.average(preds[i * tot_increase_num: ((i + 1) * tot_increase_num) - 1], axis=0))

        y_pred = np.array(y_predicted)

        keras.backend.clear_session()

        return y_pred
