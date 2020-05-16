__author__ = "Withington"

from tensorflow import keras

from sktime_dl.deeplearning.base.estimators import BaseDeepRegressor
from sktime_dl.deeplearning.lstm._base import LSTMNetwork
from sktime_dl.utils import check_and_clean_data


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
            random_seed=0,
            verbose=False,
            model_name="lstm_regressor",
            model_save_directory=None
    ):
        """
        :param nb_epochs: int, the number of epochs to train the model
        :param batch_size: int, the number of samples per gradient update.
        :param units: int, array of size 2, the number units in each LSTM layer
        :param random_seed: int, seed to any needed random actions
        """
        super(LSTMRegressor, self).__init__(
            model_save_directory=model_save_directory,
            model_name=model_name
        )
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.units = units
        self.random_seed = random_seed
        self.verbose = verbose

        self.is_fitted = False

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

    def fit(self, X, y, input_checks=True, **kwargs):
        """
        Build the regressor on the training set (X, y)
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame of Series
            objects is passed, column 0 is extracted.
        y : array-like, shape = [n_instances]
            The regression values.
        input_checks: boolean
            whether to check the X and y parameters
        Returns
        -------
        self : object
        """
        X = check_and_clean_data(X, y, input_checks=input_checks)

        self.input_shape = X.shape[1:]

        self.model = self.build_model(self.input_shape)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(
            X, y, batch_size=self.batch_size,
            epochs=self.nb_epochs, verbose=self.verbose)

        self.save_trained_model()
        self.is_fitted = True

        return self
