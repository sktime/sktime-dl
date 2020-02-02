import keras
import numpy as np

from sktime_dl.deeplearning.base.estimators._classifier import BaseDeepClassifier


class InceptionTimeClassifier(BaseDeepClassifier):
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
                 model_name="inception",
                 model_save_directory=None):
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
        self.model_name = model_name
        self.model_save_directory = model_save_directory
        self.is_fitted_ = False

        # predefined
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.bottleneck_size = bottleneck_size
        self.nb_epochs = nb_epochs

        # calced in fit
        self.classes_ = None
        self.nb_classes = -1
        self.input_shape = None
        self.model = None
        self.history = None
        self.callbacks = None

        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        """
        Construct a compiled, un-trained, keras model that is ready for training
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        nb_classes: int
            The number of classes, which shall become the size of the output layer
        Returns
        -------
        output : a compiled Keras Model
        """
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        self.callbacks = [reduce_lr]

        return model

    def fit(self, X, y, input_checks=True, **kwargs):
        """
        Build the classifier on the training set (X, y)
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame is passed, column 0 is extracted.
        y : array-like, shape = [n_instances]
            The class labels.
        input_checks: boolean
            whether to check the X and y parameters
        Returns
        -------
        self : object
        """
        X = self.check_and_clean_data(X, y, input_checks=input_checks)

        y_onehot = self.convert_y(y)
        self.input_shape = X.shape[1:]

        if self.batch_size is None:
            self.batch_size = int(min(X.shape[0] / 10, 16))
        else:
            self.batch_size = self.batch_size

        self.model = self.build_model(self.input_shape, self.nb_classes)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(X, y_onehot, batch_size=self.batch_size, epochs=self.nb_epochs,
                                      verbose=self.verbose, callbacks=self.callbacks)

        self.save_trained_model()
        self.is_fitted_ = True

        return self
