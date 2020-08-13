__author__ = "Withington"

from tensorflow import keras

from sktime_dl.deeplearning.base.estimators import BaseDeepRegressor
from sktime_dl.deeplearning.lstm._base import LSTMNetwork
from sktime_dl.utils import check_and_clean_data, \
    check_and_clean_validation_data


class LSTMRegressor(BaseDeepRegressor, LSTMNetwork):
    """ Long Short-Term Memory (LSTM)

    Adapted from the implementation of Brownlee, J. (2018)

    https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
    """

    def __init__(
            self,
            nb_epochs=200,
            batch_size=16,
            units=[50, 50],
            random_state=0,
            verbose=False,
            model_name="lstm_regressor",
            model_save_directory=None
    ):
        """
        :param nb_epochs: int, the number of epochs to train the model
        :param batch_size: int, the number of samples per gradient update.
        :param units: int, array of size 2, the number units in each LSTM layer
        :param random_state: int, seed to any needed random actions
        """
        super(LSTMRegressor, self).__init__(
            model_save_directory=model_save_directory,
            model_name=model_name
        )
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.units = units
        self.random_state = random_state
        self.verbose = verbose

        self._is_fitted = False

    def build_model(self, input_shape, **kwargs):
        """
        Construct a compiled, un-trained, keras model
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        Returns
        -------
        output : a compiled Keras Model
        """
        input_layer, output_layer = self.build_network(input_shape, **kwargs)
        output_layer = keras.layers.Dense(units=1)(output_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss='mean_squared_error',
            optimizer=keras.optimizers.Adam(),
            metrics=['mean_squared_error'])
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

        validation_data = \
            check_and_clean_validation_data(validation_X, validation_y)

        self.input_shape = X.shape[1:]

        self.model = self.build_model(self.input_shape)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            validation_data=validation_data,
        )

        self.save_trained_model()
        self._is_fitted = True

        return self
