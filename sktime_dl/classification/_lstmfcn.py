__author__ = "Jack Russon"

import numpy as np
from tensorflow import keras

from sktime_dl.classification._classifier import BaseDeepClassifier
from sktime_dl.networks._lstmfcn import LSTMFCNNetwork
from sktime_dl.utils import check_and_clean_data, \
    check_and_clean_validation_data
from sktime_dl.utils import check_is_fitted
from sklearn.utils import check_random_state


class LSTMFCNClassifier(BaseDeepClassifier, LSTMFCNNetwork):
    """
    """

    def __init__(
            self,
            nb_epochs=120,
            batch_size=16,
            kernel_sizes=[8, 5, 3],
            filter_sizes=[128, 256, 128],
            NUM_CELLS=8,
            dropout=0.8,
            attention=False,
            callbacks=[],
            random_state=0,
            verbose=False,
            model_name="lstmfcn",
            model_save_directory=None,
    ):
        """
        :param nb_epochs: int, the number of epochs to train the model
        :param batch_size: int, the number of samples per gradient update.
        :param kernel_sizes: list of ints, specifying the length of the 1D convolution
         windows
        :param filter_sizes: int, array of shape = 3, size of filter for each
         conv layer
        :param callbacks: not used
        :param random_state: int, seed to any needed random actions
        :param verbose: boolean, whether to output extra information
        :param model_name: string, the name of this model for printing and
        file writing purposes
        :param model_save_directory: string, if not None; location to save
        the trained keras model in hdf5 format
        """
        super(LSTMFCNClassifier, self).__init__(
            model_name=model_name, model_save_directory=model_save_directory
        )

        self.verbose = verbose
        self._is_fitted = False

        # calced in fit
        self.classes_ = None
        self.nb_classes = -1
        self.input_shape = None
        self.model = None
        self.history = None

        # predefined
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.kernel_sizes = kernel_sizes
        self.filter_sizes = filter_sizes
        self.NUM_CELLS=NUM_CELLS
        self.dropout=dropout
        self.attention=attention

        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose

        self._is_fitted = False

    def build_model(self, input_shape, nb_classes, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for
        training
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        nb_classes: int
            The number of classes, which shall become the size of the output
             layer
        Returns
        -------
        output : a compiled Keras Model
        """
        input_layers, output_layer = self.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(nb_classes, activation="softmax")(
            output_layer
        )

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.SGD(
                lr=0.01, momentum=0.9, decay=0.0005
            ),
            metrics=["accuracy"],
        )

        # file_path = self.output_directory + 'best_model.hdf5'
        # model_checkpoint = keras.callbacks.ModelCheckpoint(
        #     filepath=file_path, monitor='val_loss',
        #     save_best_only=True)
        # self.callbacks = [model_checkpoint]
        self.callbacks = []

        return model

    def fit(self, X, y, input_checks=True, validation_X=None,
            validation_y=None, **kwargs):
        """
        Fit the classifier on the training set (X, y)
        ----------
        X : a nested pd.Dataframe, or (if input_checks=False) array-like of
        shape = (n_instances, series_length, n_dimensions)
            The training input samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
        y : array-like, shape = [n_instances]
            The training data class labels.
        input_checks : boolean
            whether to check the X and y parameters
        validation_X : a nested pd.Dataframe, or array-like of shape =
        (n_instances, series_length, n_dimensions)
            The validation samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
            Unless strictly defined by the user via callbacks (such as
            EarlyStopping), the presence or state of the validation
            data does not alter training in any way. Predictions at each epoch
            are stored in the model's fit history.
        validation_y : array-like, shape = [n_instances]
            The validation class labels.
        Returns
        -------
        self : object
        """
        self.random_state = check_random_state(self.random_state)

        X = check_and_clean_data(X, y, input_checks=input_checks)
        y_onehot = self.convert_y(y)

        validation_data = \
            check_and_clean_validation_data(validation_X, validation_y,
                                            self.label_encoder,
                                            self.onehot_encoder)

        # ignore the number of instances, X.shape[0],
        # just want the shape of each instance
        self.input_shape = X.shape[1:]


        if validation_data is not None:
            validation_data = (
                self.prepare_input(validation_data[0]),
                validation_data[1]
            )

        self.model = self.build_model(self.input_shape, self.nb_classes)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(
            X,
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            validation_data=(validation_data),
            callbacks=self.callbacks,
        )

        self.save_trained_model()
        self._is_fitted = True

        return self

    def predict_proba(self, X, input_checks=True, **kwargs):
        """
        Find probability estimates for each class for all cases in X.
        Parameters
        ----------
        X : a nested pd.Dataframe, or (if input_checks=False) array-like of
        shape = (n_instances, series_length, n_dimensions)
            The training input samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
        input_checks: boolean
            whether to check the X parameter
        Returns
        -------
        output : array of shape = [n_instances, n_classes] of probabilities
        """
        check_is_fitted(self)

        X = check_and_clean_data(X, input_checks=input_checks)



        probs = self.model.predict(X, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])

        return probs
