__author__ = "James Large, Withington"

from tensorflow import keras

from sktime_dl.deeplearning.base.estimators import BaseDeepRegressor
from sktime_dl.deeplearning.cnn._base import CNNNetwork
from sktime_dl.utils import check_and_clean_data, \
    check_and_clean_validation_data


class CNNRegressor(BaseDeepRegressor, CNNNetwork):
    """Time Convolutional Neural Network (CNN).

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/cnn.py

    Network originally defined in:

    @article{zhao2017convolutional, title={Convolutional neural networks for
    time series classification}, author={Zhao, Bendong and Lu, Huanzhang and
    Chen, Shangfeng and Liu, Junliang and Wu, Dongya}, journal={Journal of
    Systems Engineering and Electronics}, volume={28}, number={1}, pages={
    162--169}, year={2017}, publisher={BIAI} }

    :param nb_epochs: int, the number of epochs to train the model
    :param batch_size: int, the number of samples per gradient update.
    :param kernel_size: int, specifying the length of the 1D convolution
     window
    :param avg_pool_size: int, size of the average pooling windows
    :param nb_conv_layers: int, the number of convolutional plus average
     pooling layers
    :param filter_sizes: int, array of shape = (nb_conv_layers)
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
            nb_epochs=2000,
            batch_size=16,
            kernel_size=7,
            avg_pool_size=3,
            nb_conv_layers=2,
            filter_sizes=[6, 12],
            callbacks=None,
            random_state=0,
            verbose=False,
            model_name="cnn_regressor",
            model_save_directory=None,
    ):
        super(CNNRegressor, self).__init__(
            model_save_directory=model_save_directory,
            model_name=model_name,
        )
        self.filter_sizes = filter_sizes
        self.nb_conv_layers = nb_conv_layers
        self.avg_pool_size = avg_pool_size
        self.random_state = random_state
        self.kernel_size = kernel_size
        self.verbose = verbose
        self.callbacks = callbacks
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size

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

        validation_data = \
            check_and_clean_validation_data(validation_X, validation_y)

        # ignore the number of instances, X.shape[0],
        # just want the shape of each instance
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
            callbacks=self.callbacks,
            validation_data=validation_data,
        )

        self.save_trained_model()
        self._is_fitted = True

        return self
