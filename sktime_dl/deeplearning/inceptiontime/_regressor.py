__author__ = "James Large, Withington"

from tensorflow import keras

from sktime_dl.deeplearning.base.estimators import BaseDeepRegressor
from sktime_dl.deeplearning.inceptiontime._base import InceptionTimeNetwork


class InceptionTimeRegressor(BaseDeepRegressor, InceptionTimeNetwork):
    """InceptionTime

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py

    Network originally defined in:

    @article{IsmailFawaz2019inceptionTime,
      Title                    = {InceptionTime: Finding AlexNet for Time Series Classification},
      Author                   = {Ismail Fawaz, Hassan and Lucas, Benjamin and Forestier, Germain and Pelletier, Charlotte and Schmidt, Daniel F. and Weber, Jonathan and Webb, Geoffrey I. and Idoumghar, Lhassane and Muller, Pierre-Alain and Petitjean, François},
      journal                  = {ArXiv},
      Year                     = {2019}
    }
    """

    def __init__(self,
                 nb_filters=32,
                 use_residual=True,
                 use_bottleneck=True,
                 bottleneck_size=32,
                 depth=6,
                 kernel_size=41 - 1,
                 batch_size=64,
                 nb_epochs=1500,
                 random_seed=0,
                 verbose=False,
                 model_name="inception_regressor",
                 model_save_directory=None):
        super().__init__(
            model_name=model_name,
            model_save_directory=model_save_directory)
        InceptionTimeNetwork.__init__(
            self,
            nb_filters=nb_filters,
            use_residual=use_residual,
            use_bottleneck=use_bottleneck,
            bottleneck_size=bottleneck_size,
            depth=depth,
            kernel_size=kernel_size,
            random_seed=random_seed)
        '''
        :param nb_filters: int,
        :param use_residual: boolean,
        :param use_bottleneck: boolean,
        :param depth: int
        :param kernel_size: int, specifying the length of the 1D convolution window
        :param batch_size: int, the number of samples per gradient update.
        :param bottleneck_size: int,
        :param nb_epochs: int, the number of epochs to train the model
        :param random_seed: int, seed to any needed random actions
        :param verbose: boolean, whether to output extra information
        :param model_name: string, the name of this model for printing and file writing purposes
        :param model_save_directory: string, if not None; location to save the trained keras model in hdf5 format
        '''

        self.verbose = verbose
        self.is_fitted_ = False

        # predefined
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs

        # calced in fit
        self.input_shape = None
        self.history = None
        self.callbacks = None

    def build_model(self, input_shape, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for training
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
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
        self.callbacks = [reduce_lr]

        return model

    def fit(self, X, y, input_checks=True, **kwargs):
        """
        Build the regressor on the training set (X, y)
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame of Series objects is passed, column 0 is extracted.
        y : array-like, shape = [n_instances]
            The regression values.
        input_checks: boolean
            whether to check the X and y parameters
        Returns
        -------
        self : object
        """
        X = self.check_and_clean_data(X, y, input_checks=input_checks)

        self.input_shape = X.shape[1:]

        if self.batch_size is None:
            self.batch_size = int(max(1, min(X.shape[0] / 10, 16)))
        else:
            self.batch_size = self.batch_size

        self.model = self.build_model(self.input_shape)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(X, y, batch_size=self.batch_size, epochs=self.nb_epochs,
                                      verbose=self.verbose, callbacks=self.callbacks)

        self.save_trained_model()
        self.is_fitted_ = True

        return self
