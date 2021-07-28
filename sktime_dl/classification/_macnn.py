__author__ = "Jack Russon"

from sktime_dl.classification._classifier import BaseDeepClassifier
from sktime_dl.networks._macnn import MACNNNetwork
from sktime_dl.utils import check_and_clean_data, \
    check_and_clean_validation_data
from sklearn.utils import check_random_state
from tensorflow import keras


class MACNNClassifier(BaseDeepClassifier, MACNNNetwork):

    def __init__(
            self,
            nb_epochs=1500,
            batch_size=4,
            callbacks=[],
            random_state=0,
            verbose=False,
            model_name="macnn",
            model_save_directory=None,
    ):
        """
        :param nb_epochs: int, the number of epochs to train the model
        :param batch_size: int, the number of samples per gradient update.
        :param rnn_layer: int, filter size for rnn layer
        :param filter_sizes: int, array of shape 2, filter sizes for two convolutional layers
        :param kernel_sizes: int,array of shape 2,  kernel size for two convolutional layers
        :param lstm_size: int, filter size of lstm layer
        :param dense_size: int, size of dense layer
        :param callbacks: not used
        :param random_state: int, seed to any needed random actions
        :param verbose: boolean, whether to output extra information
        :param model_name: string, the name of this model for printing and
        file writing purposes
        :param model_save_directory: string, if not None; location to save
        the trained keras model in hdf5 format
        """
        super(MACNNClassifier, self).__init__(
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
        self.random_state = random_state


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
        input_layer, output_layer = self.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(
            units=nb_classes, activation="softmax"
        )(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(lr=0.0001),
            metrics=["accuracy"]
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.7,
                                                      patience=50, min_lr=0.00001)
        self.callbacks = [reduce_lr]

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
            whether to check the X and y paramete
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

        if self.callbacks is None:
            self.callbacks = []

        X = check_and_clean_data(X, y, input_checks=input_checks)
        y_onehot = self.convert_y(y)

        validation_data = \
            check_and_clean_validation_data(validation_X, validation_y,
                                            self.label_encoder,
                                            self.onehot_encoder)

        # ignore the number of instances, X.shape[0],
        # just want the shape of each instance
        self.input_shape = X.shape[1:]

        self.model = self.build_model(self.input_shape, self.nb_classes)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(
            X,
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
            validation_data=validation_data
        )

        self._is_fitted = True
        self.save_trained_model()

        return self