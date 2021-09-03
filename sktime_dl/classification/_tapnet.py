# -*- coding: utf-8 -*-
"""Time Convolutional Neural Network (CNN) for classification"""


__author__ = "Jack Russon"

from sktime_dl.classification._classifier import BaseDeepClassifier
from sktime_dl.networks._tapnet import TapNetNetwork
from sktime_dl.utils import check_and_clean_data, \
    check_and_clean_validation_data
from sklearn.utils import check_random_state
from tensorflow import keras


class TapNetClassifier(BaseDeepClassifier, TapNetNetwork):
    """
    Implementation of TapNetClassifier Zhang (2020). [1]_
    Overview:
    Uses optionally an LSTM, CNN and self attention mechanism. Original implementation used pytorch
    so some features have not been added.


    Parameters
    ----------

    filter_sizes: list or array of ints, default=[256, 256, 128]
        sets the kernel size argument for each convolutional block. Controls number of convolutional filters
        and number of neurons in attention dense layers.
    kernel_size : list or array of ints, default=[8, 5, 3]
        controls the size of the convolutional kernels
    use_rp: boolean, default=True
        Whether to use random projections or not
    use_att: boolean, default=True
        Use self attention in the model
    use_cnn: boolean, default=True
        Use cnn layers
    use_lstm: boolean, default=True
        Use lstm layers
    layers: list of ints, default=[500,300]
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

    --------

    [1] @inproceedings{zhang2020tapnet,
    title={Tapnet: Multivariate time series classification with attentional prototypical network},
    author={Zhang, Xuchao and Gao, Yifeng and Lin, Jessica and Lu, Chang-Tien},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={34},
    number={04},
    pages={6845--6852},
    year={2020}
    }

    Example
    -------
    from sktime_dl.classification import TapNetClassifier
    from sktime.datasets import load_italy_power_demand
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    clf = TapNetClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    """

    def __init__(
            self,
            nb_epochs=2000,
            batch_size=16,
            dropout=0.5,
            filter_sizes=[256, 256, 128],
            kernel_sizes=[8, 5, 3],
            dilation=1,
            layers=[500, 300],
            use_rp=True,
            rp_params=[-1, 3],
            use_att=True,
            use_lstm=True,
            use_cnn=True,
            random_state=1,
            padding='same',
            callbacks=None,
            verbose=False,
            model_name="tapnet",
            model_save_directory=None

    ):
        """
        :param kernel_size: int, specifying the length of the 1D convolution
         window
        :param avg_pool_size: int, size of the average pooling windows
        :param layers: int, size of dense layers
        :param filter_sizes: int, array of shape = (nb_conv_layers)
        :param random_state: int, seed to any needed random actions
        :param rp_params: array of ints, parameters for random permutation
        :param dropout: dropout rate
        """
        super(TapNetClassifier, self).__init__(
            model_save_directory=model_save_directory,
            model_name=model_name)
        self.batch_size=batch_size
        self.random_state = random_state
        self.kernel_size = kernel_sizes
        self.layers = layers
        self.rp_params = rp_params
        self.filter_sizes = filter_sizes
        self.use_att = use_att

        self.dilation = dilation
        self.padding = padding
        self.nb_epochs=nb_epochs
        self.callbacks=callbacks
        self.verbose=verbose

        self._is_fitted=False

        self.dropout = dropout
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn

        # parameters for random projection
        self.use_rp = use_rp
        self.rp_params = rp_params

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
            optimizer=keras.optimizers.Adam(learning_rate=0.00001),
            metrics=["accuracy"]
        )
        #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.7,
         #                                             patience=50, min_lr=0.0001)
        # if self.callbacks==None:
        #
        #     self.callbacks.append = (reduce_lr)

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


