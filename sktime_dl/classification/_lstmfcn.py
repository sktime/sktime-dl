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

    Implementation of LSTMFCNClassifier from Karim et al (2019). [1]_
    Overview:
     Combines an LSTM arm with a CNN arm. Optionally uses an attention mechanism in the LSTM which the
     author indicates provides improved performance.


    Parameters
    ----------
    nb_epochs: int, default=1500
     the number of epochs to train the model
    param batch_size: int, default=128
        the number of samples per gradient update.
    kernel_sizes: list of ints, default=[8, 5, 3]
        specifying the length of the 1D convolution windows
    filter_sizes: int, list of ints, default=[128, 256, 128]
        size of filter for each conv layer
    num_cells: int, default=8
        output dimension for LSTM layer
    dropout: float, default=0.8
        controls dropout rate of LSTM layer
    attention: boolean, default=False
        If True, uses custom attention LSTM layer
    callbacks: keras callbacks, default=ReduceLRonPlateau
        Keras callbacks to use such as learning rate reduction or saving best model based on validation error
    random_state: int,
        seed to any needed random actions
    verbose: boolean,
        whether to output extra information
    model_name: string,
        the name of this model for printing and file writing purposes
    model_save_directory: string,
        if not None; location to save the trained keras model in hdf5 format
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random, integer.
    Attributes
    ----------
    nb_classes : int
        Number of classes. Extracted from the data.

    References
    ----------
    @article{Karim_2019,
    title={Multivariate LSTM-FCNs for time series classification},
    volume={116},
    ISSN={0893-6080},
    url={http://dx.doi.org/10.1016/j.neunet.2019.04.014},
    DOI={10.1016/j.neunet.2019.04.014},
    journal={Neural Networks},
    publisher={Elsevier BV},
    author={Karim, Fazle and Majumdar, Somshubra and Darabi, Houshang and Harford, Samuel},
    year={2019},
    month={Aug},
    pages={237–245}
    }



    Example
    -------
    from sktime_dl.classification import LSTMFCNClassifier
    from sktime.datasets import load_italy_power_demand
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    clf = LSTMFCNClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    """

    def __init__(
            self,
            nb_epochs=2000,
            batch_size=128,
            dropout=0.8,
            kernel_sizes=[8, 5, 3],
            filter_sizes=[128, 256, 128],
            lstm_size=8,
            use_att=False,
            callbacks=None,
            random_state=0,
            verbose=False,
            model_name="lstmfcn",
            model_save_directory=None,
    ):

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
        self.NUM_CELLS=lstm_size
        self.dropout=dropout
        self.attention=use_att

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
            optimizer='adam',
            metrics=["accuracy"],
        )

        # file_path = self.output_directory + 'best_model.hdf5'
        # model_checkpoint = keras.callbacks.ModelCheckpoint(
        #     filepath=file_path, monitor='val_loss',
        #     save_best_only=True)
        # self.callbacks = [model_checkpoint]
        if self.callbacks==None:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.7,
                                                          patience=50, min_lr=0.0001)
            self.callbacks = [reduce_lr]
        else:
            pass

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
                validation_data[0],
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
