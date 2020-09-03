__author__ = "James Large, Withington"

import numpy as np
from tensorflow import keras

from sktime_dl.deeplearning.base.estimators import BaseDeepRegressor
from sktime_dl.deeplearning.tlenet._base import TLENETNetwork
from sktime_dl.utils import check_and_clean_data, \
    check_and_clean_validation_data, check_is_fitted


class TLENETRegressor(BaseDeepRegressor, TLENETNetwork):
    """Time Le-Net (TLENET).

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/tlenet.py

    Network originally defined in:

    @inproceedings{le2016data, title={Data augmentation for time series
    classification using convolutional neural networks}, author={Le Guennec,
    Arthur and Malinowski, Simon and Tavenard, Romain}, booktitle={ECML/PKDD
    workshop on advanced analytics and learning on temporal data},
    year={2016} }
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
            model_name="tlenet_regressor",
            model_save_directory=None
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
        super(TLENETRegressor, self).__init__(
            model_name=model_name, model_save_directory=model_save_directory)

        self.verbose = verbose
        self._is_fitted = False

        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.warping_ratios = warping_ratios
        self.slice_ratio = slice_ratio

        self.callbacks = callbacks
        self.verbose = verbose
        self.random_state = random_state

        self._is_fitted = False

    def build_model(self, input_shape, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for
         training
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        Returns
        -------
        output : a compiled Keras Model
        """
        save_best_model = False

        input_layer, output_layer = self.build_network(input_shape, **kwargs)
        output_layer = keras.layers.Dense(units=1)(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            optimizer=keras.optimizers.Adam(lr=0.01, decay=0.005),
            loss="mean_squared_error",
            metrics=["mean_squared_error"],
        )

        if self.callbacks is None:
            self.callbacks = []

        if save_best_model:
            file_path = self.model_save_directory + "best_model.hdf5"
            model_checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=file_path, monitor="loss", save_best_only=True
            )
            self.callbacks.append(model_checkpoint)

        return model

    def fit(self, X, y, input_checks=True, validation_X=None,
            validation_y=None, **kwargs):
        """
        Fit the regressor on the training set (X, y)
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
        X = check_and_clean_data(X, y, input_checks=input_checks)

        self.adjust_parameters(X)
        X, y, __ = self.pre_processing(X, y)

        validation_data = \
            check_and_clean_validation_data(validation_X, validation_y)
        if validation_data is not None:
            vX, vy, _ = self.pre_processing(validation_data[0],
                                            validation_data[1])
            validation_data = (vX, vy)

        input_shape = X.shape[1:]
        self.model = self.build_model(input_shape)

        self.history = self.model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
            validation_data=validation_data,
        )

        self.save_trained_model()
        self._is_fitted = True

        return self

    def predict(self, X, input_checks=True, **kwargs):
        """
        Find regression estimate for all cases in X.
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
        predictions : 1d numpy array
            array of predictions of each instance
        """
        check_is_fitted(self)

        X = check_and_clean_data(X, input_checks=input_checks)

        X, _, tot_increase_num = self.pre_processing(X)

        preds = self.model.predict(X, batch_size=self.batch_size)

        y_predicted = []
        test_num_batch = int(X.shape[0] / tot_increase_num)

        # TODO: could fix this to be an array literal.
        for i in range(test_num_batch):
            y_predicted.append(
                np.average(preds[i
                                 * tot_increase_num: ((i + 1)
                                                      * tot_increase_num) - 1],
                           axis=0,
                           )
            )

        y_pred = np.array(y_predicted)

        if y_pred.ndim == 1:
            y_pred.ravel()

        return y_pred
