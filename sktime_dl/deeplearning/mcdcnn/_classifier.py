__author__ = "James Large"

from tensorflow import keras
import numpy as np

from sklearn.model_selection import train_test_split

from sktime_dl.deeplearning.base.estimators import BaseDeepClassifier


class MCDCNNClassifier(BaseDeepClassifier):
    """Multi Channel Deep Convolutional Neural Network (MCDCNN).

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mcdcnn.py

    Network originally defined in:

    @inproceedings{zheng2014time,
      title={Time series classification using multi-channels deep convolutional neural networks},
      author={Zheng, Yi and Liu, Qi and Chen, Enhong and Ge, Yong and Zhao, J Leon},
      booktitle={International Conference on Web-Age Information Management},
      pages={298--310},
      year={2014},
      organization={Springer}
    }
    """

    def __init__(self,
                 nb_epochs=120,
                 batch_size=16,

                 callbacks=[],
                 random_seed=0,
                 verbose=False,
                 model_name="mcdcnn",
                 model_save_directory=None):
        '''
        :param nb_epochs: int, the number of epochs to train the model
        :param batch_size: int, the number of samples per gradient update.
        :param random_seed: int, seed to any needed random actions
        :param verbose: boolean, whether to output extra information
        :param model_name: string, the name of this model for printing and file writing purposes
        :param model_save_directory: string, if not None; location to save the trained keras model in hdf5 format
        '''

        self.verbose = verbose
        self.model_name = model_name
        self.model_save_directory = model_save_directory
        self.is_fitted_ = False

        # calced in fit
        self.classes_ = None
        self.nb_classes = -1
        self.input_shape = None
        self.model = None
        self.history = None

        # predefined
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.callbacks = callbacks

        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)

    def build_model(self, input_shape, nb_classes, **kwargs):
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
        n_t = input_shape[0]
        n_vars = input_shape[1]

        padding = 'valid'

        if n_t < 60:  # for ItalyPowerOndemand
            padding = 'same'

        input_layers = []
        conv2_layers = []

        for n_var in range(n_vars):
            input_layer = keras.layers.Input((n_t, 1))
            input_layers.append(input_layer)

            conv1_layer = keras.layers.Conv1D(filters=8, kernel_size=5, activation='relu', padding=padding)(input_layer)
            conv1_layer = keras.layers.MaxPooling1D(pool_size=2)(conv1_layer)

            conv2_layer = keras.layers.Conv1D(filters=8, kernel_size=5, activation='relu', padding=padding)(conv1_layer)
            conv2_layer = keras.layers.MaxPooling1D(pool_size=2)(conv2_layer)
            conv2_layer = keras.layers.Flatten()(conv2_layer)

            conv2_layers.append(conv2_layer)

        if n_vars == 1:
            # to work with univariate time series
            concat_layer = conv2_layers[0]
        else:
            concat_layer = keras.layers.Concatenate(axis=-1)(conv2_layers)

        fully_connected = keras.layers.Dense(units=732, activation='relu')(concat_layer)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(fully_connected)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0005),
                      metrics=['accuracy'])

        # file_path = self.output_directory + 'best_model.hdf5'
        # model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
        #                                                   save_best_only=True)
        # self.callbacks = [model_checkpoint]
        self.callbacks = []

        return model

    def prepare_input(self, x):
        new_x = []
        n_t = x.shape[1]
        n_vars = x.shape[2]

        for i in range(n_vars):
            new_x.append(x[:, :, i:i + 1])

        return new_x

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
        x_train, x_val, y_train_onehot, y_val_onehot = \
            train_test_split(X, y_onehot, test_size=0.33)

        x_train = self.prepare_input(x_train)
        x_val = self.prepare_input(x_val)

        self.input_shape = X.shape[1:]

        self.model = self.build_model(self.input_shape, self.nb_classes)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(x_train, y_train_onehot, batch_size=self.batch_size, epochs=self.nb_epochs,
                                      verbose=self.verbose, validation_data=(x_val, y_val_onehot),
                                      callbacks=self.callbacks)

        self.save_trained_model()
        self.is_fitted_ = True

        return self

    def predict_proba(self, X, input_checks=True, **kwargs):
        """
        Find probability estimates for each class for all cases in X.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.
            If a Pandas data frame is passed (sktime format)
            If a Pandas data frame is passed, a check is performed that it only has one column.
            If not, an exception is thrown, since this classifier does not yet have
            multivariate capability.
        input_checks: boolean
            whether to check the X parameter
        Returns
        -------
        output : array of shape = [n_instances, n_classes] of probabilities
        """
        X = self.check_and_clean_data(X, input_checks=input_checks)

        x_test = self.prepare_input(X)

        probs = self.model.predict(x_test, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])

        return probs
