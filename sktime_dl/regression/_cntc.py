__author__ = "James Large, Withington"

from tensorflow import keras

from sktime_dl.regression._regressor import BaseDeepRegressor
from sktime_dl.networks._cntc import CNTCNetwork
from sktime_dl.utils import check_and_clean_data, \
    check_and_clean_validation_data


class CNTCRegressor(BaseDeepRegressor, CNTCNetwork):
    """Combining contextual neural networks for time series classification

       Adapted from the implementation from Fullah et. al

      https://github.com/AmaduFullah/CNTC_MODEL/blob/master/cntc.ipynb

       Network originally defined in:

       @article{FULLAHKAMARA202057,
       title = {Combining contextual neural networks for time series classification},
       journal = {Neurocomputing},
       volume = {384},
       pages = {57-66},
       year = {2020},
       issn = {0925-2312},
       doi = {https://doi.org/10.1016/j.neucom.2019.10.113},
       url = {https://www.sciencedirect.com/science/article/pii/S0925231219316364},
       author = {Amadu {Fullah Kamara} and Enhong Chen and Qi Liu and Zhen Pan},
       keywords = {Time series classification, Contextual convolutional neural networks, Contextual long short-term memory, Attention, Multilayer perceptron},
      }

    :param nb_epochs: int, the number of epochs to train the model
    :param batch_size: int, the number of samples per gradient update.
    :param rnn_layer: int, filter size for rnn layer
    :param filter_sizes: int, array of shape 2, filter sizes for two convolutional layers
    :param kernel_sizes: int,array of shape 2,  kernel size for two convolutional layers
    :param lstm_size: int, filter size of lstm layer
    :param dense_size: int, size of dense layer
    :param callbacks: list of tf.keras.callbacks.Callback objects
    :param random_state: int, seed to any needed random actions
    :param verbose: boolean, whether to output extra information
    :param model_name: string, the name of this model for printing and file
     writing purposes
    :param model_save_directory: string, if not None; location to save the
     trained keras model in hdf5 format
    """

    def __init__(
            self,
            nb_epochs=120,
            batch_size=64,
            rnn_layer=64,
            filter_sizes=[16, 8],
            kernel_sizes=[1, 1],
            lstm_size=8,
            dense_size=64,
            callbacks=None,
            random_state=0,
            verbose=False,
            model_name="cntc_regressor",
            model_save_directory=None,
    ):
        super(CNTCRegressor, self).__init__(
            model_save_directory=model_save_directory,
            model_name=model_name,
        )
        self.random_state = random_state
        self.verbose = verbose
        self.callbacks = callbacks
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.rnn_layer = rnn_layer
        self.filter_sizes = filter_sizes
        self.kernel_sizes = kernel_sizes
        self.lstm_size = lstm_size
        self.dense_size = dense_size

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
        input_layer, output_layer = self.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(units=1)(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss="mean_squared_error",
            optimizer=keras.optimizers.Adam(),
            metrics=["mean_squared_error"],
        )

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
        if self.callbacks is None:
            self.callbacks = []

        X = check_and_clean_data(X, y, input_checks=input_checks)
        X_2=self.prepare_input(X)
        validation_data = \
            check_and_clean_validation_data(validation_X, validation_y)

        # ignore the number of instances, X.shape[0],
        # just want the shape of each instance
        self.input_shape = X.shape[1:]

        self.model = self.build_model(self.input_shape)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(
            [X_2,X,X],
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
        self.check_is_fitted()

        X = check_and_clean_data(X, input_checks=input_checks)

        x_test= self.prepare_input(X)

        preds = self.model.predict([x_test,X,X], batch_size=self.batch_size)

        return preds



