__author__ = "Jack Russon"

from sktime_dl.classification._classifier import BaseDeepClassifier
from sktime_dl.networks._macnn import MACNNNetwork
from sktime_dl.utils import check_and_clean_data, \
    check_and_clean_validation_data
from sklearn.utils import check_random_state
from tensorflow import keras


class MACNNClassifier(BaseDeepClassifier, MACNNNetwork):
    """

    Implementation of MACNNClassifier from Chen (2021). [1]_
    Overview:
     Neural Network made of multiple convolutional attention blocks. The block is separated into three
     sections. Section 1 and 2 are made up of two blocks followed by a max pooling layer. The final section
     contains two blocks and a mean reduction followed by the dense output layer.


    Parameters
    ----------
    pool_size : int, default=3
       Controls pool size for maxpooling layers.
    stride : int, default=2
       Controls stride length for maxpooling layers.
    filters: list or array of ints, default=[64, 128, 256]
        sets the kernel size argument for each convolutional block. Controls number of convolutional filters
        and number of neurons in attention dense layers.
    kernel_size : list or array of ints, default=[3, 6, 12]
        controls the size of the convolutional kernels
    reduction : int, default=16
        divides the number of dense neurons in the first layer of the attention block.
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
    [1] Wei Chen, Ke Shi,
    Multi-scale Attention Convolutional Neural Network for time series classification,
    Neural Networks,
    Volume 136,
    2021,
    Pages 126-140,
    ISSN 0893-6080,
    https://doi.org/10.1016/j.neunet.2021.01.001.
    (https://www.sciencedirect.com/science/article/pii/S0893608021000010)

    Example
    -------
    from sktime_dl.classification import MACNNClassifier
    from sktime.datasets import load_italy_power_demand
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    clf = MACNNClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    """

    def __init__(
            self,
            nb_epochs=1500,
            batch_size=4,
            padding='same',
            max_pool_size=3,
            stride=2,
            repeats=2,
            filter_sizes=[64, 128, 256],
            kernel_sizes=[3, 6, 12],
            reduction=16,
            callbacks=None,
            random_state=0,
            verbose=False,
            model_name="macnn",
            model_save_directory=None,
    ):
        """
        :param nb_epochs: int, the number of epochs to train the model
        :param batch_size: int, the number of samples per gradient update.
        :param rnn_layer: int, filter size for rnn layer
        :param filters: int, array of shape 2, filter sizes for two convolutional layers
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

        self.kernel_sizes = kernel_sizes
        self.filters= filter_sizes
        self.reduction = reduction
        self.pool_size = max_pool_size
        self.stride = stride
        self.repeats = repeats
        self.padding=padding

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