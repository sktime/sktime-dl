__author__ = "Aaron Bostrom, James Large"

import numpy as np

from sktime_dl.deeplearning.base.estimators import BaseDeepClassifier
from sktime_dl.deeplearning.tlenet._base import TLENETNetwork
from sktime_dl.utils import check_and_clean_data, \
    check_and_clean_validation_data, check_is_fitted
from tensorflow import keras
from sklearn.utils import check_random_state


class TLENETClassifier(BaseDeepClassifier, TLENETNetwork):
    """Time Le-Net (TLENET).

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/tlenet.py

    Network originally defined in:

    @inproceedings{le2016data,
      title={Data augmentation for time series classification using
      convolutional neural networks},
      author={Le Guennec, Arthur and Malinowski, Simon and Tavenard, Romain},
      booktitle={ECML/PKDD workshop on advanced analytics and learning on
      temporal data},
      year={2016}
    }
    """

    def __init__(
            self,
            nb_epochs=1000,
            batch_size=256,
            warping_ratios=[0.5, 1, 2],
            slice_ratio=0.1,
            callbacks=None,
            verbose=False,
            random_state=0,
            model_name="tlenet",
            model_save_directory=None,
    ):
        """
        :param nb_epochs: int, the number of epochs to train the model
        :param batch_size: int, specifying the length of the 1D convolution
         window
        :param warping_ratios: list of floats, warping ratio for each window
        :param slice_ratio: float, ratio of the time series used to create a
         slice
        :param callbacks: list of tf.keras.callbacks.Callback objects
        :param random_state: int, seed to any needed random actions
        :param verbose: boolean, whether to output extra information
        :param model_name: string, the name of this model for printing and
         file writing purposes
        :param model_save_directory: string, if not None; location to save
         the trained keras model in hdf5 format
        """
        super(TLENETClassifier, self).__init__(
            model_name=model_name, model_save_directory=model_save_directory
        )
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.warping_ratios = warping_ratios
        self.slice_ratio = slice_ratio

        self.callbacks = callbacks
        self.verbose = verbose
        self.random_state = random_state

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
        input_layer, output_layer = self.build_network(input_shape, **kwargs)
        output_layer = keras.layers.Dense(nb_classes, activation="softmax")(
            output_layer
        )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            optimizer=keras.optimizers.Adam(lr=0.01, decay=0.005),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

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
        if self.callbacks is None:
            self.callbacks = []

        self.random_state = check_random_state(self.random_state)

        X = check_and_clean_data(X, y, input_checks=input_checks)
        y_onehot = self.convert_y(y)

        # ignore the number of instances, X.shape[0],
        # just want the shape of each instance
        self.input_shape = X.shape[1:]
        self.nb_classes = y_onehot.shape[1]

        self.adjust_parameters(X)
        X, y_onehot, _ = self.pre_processing(X, y_onehot)

        validation_data = \
            check_and_clean_validation_data(validation_X, validation_y,
                                            self.label_encoder,
                                            self.onehot_encoder)
        if validation_data is not None:
            vX, vy, _ = self.pre_processing(validation_data[0],
                                            validation_data[1])
            validation_data = (vX, vy)

        input_shape = X.shape[1:]
        self.model = self.build_model(input_shape, self.nb_classes)

        self.history = self.model.fit(
            X,
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
            validation_data=validation_data,
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

        X, _, tot_increase_num = self.pre_processing(X)

        preds = self.model.predict(X, batch_size=self.batch_size)

        test_num_batch = int(X.shape[0] / tot_increase_num)

        y_predicted = [
            np.average(
                preds[i * tot_increase_num:((i + 1) * tot_increase_num) - 1],
                axis=0)
            for i in range(test_num_batch)
        ]

        y_pred = np.array(y_predicted)

        keras.backend.clear_session()

        return y_pred
