__author__ = "James Large"

from tensorflow import keras

from sktime_dl.deeplearning.base.estimators import BaseDeepClassifier
from sktime_dl.deeplearning.cnn._base import CNNNetwork
from sktime_dl.utils import check_and_clean_data


class CNNClassifier(BaseDeepClassifier, CNNNetwork):
    """Time Convolutional Neural Network (CNN).

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/cnn.py

    Network originally defined in:

    @article{zhao2017convolutional,
      title={Convolutional neural networks for time series classification},
      author={Zhao, Bendong and Lu, Huanzhang and Chen, Shangfeng and Liu, Junliang and Wu, Dongya},
      journal={Journal of Systems Engineering and Electronics},
      volume={28},
      number={1},
      pages={162--169},
      year={2017},
      publisher={BIAI}
    }
    """

    def __init__(self,
                 nb_epochs=2000,
                 batch_size=16,
                 kernel_size=7,
                 avg_pool_size=3,
                 nb_conv_layers=2,
                 filter_sizes=[6, 12],

                 callbacks=None,
                 random_seed=0,
                 verbose=False,
                 model_name="cnn",
                 model_save_directory=None):
        super().__init__(
            model_name=model_name,
            model_save_directory=model_save_directory)
        CNNNetwork.__init__(
            self,
            kernel_size=kernel_size,
            avg_pool_size=avg_pool_size,
            nb_conv_layers=nb_conv_layers,
            filter_sizes=filter_sizes,
            random_seed=random_seed)
        '''
        :param nb_epochs: int, the number of epochs to train the model
        :param batch_size: int, the number of samples per gradient update.
        :param kernel_size: int, specifying the length of the 1D convolution window
        :param avg_pool_size: int, size of the average pooling windows
        :param nb_conv_layers: int, the number of convolutional plus average pooling layers
        :param filter_sizes: int, array of shape = (nb_conv_layers)
        :param callbacks: list of tf.keras.callbacks.Callback objects
        :param random_seed: int, seed to any needed random actions
        :param verbose: boolean, whether to output extra information
        :param model_name: string, the name of this model for printing and file writing purposes
        :param model_save_directory: string, if not None; location to save the trained keras model in hdf5 format
        '''
        self.verbose = verbose
        self.is_fitted = False

        self.callbacks = callbacks if callbacks is not None else []

        self.input_shape = None
        self.history = None

        self.nb_epochs = nb_epochs
        self.batch_size = batch_size

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
        input_layer, output_layer = self.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(units=nb_classes, activation='sigmoid')(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

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
        X = check_and_clean_data(X, y, input_checks=input_checks)
        y_onehot = self.convert_y(y)

        # ignore the number of instances, X.shape[0], just want the shape of each instance
        self.input_shape = X.shape[1:]

        self.model = self.build_model(self.input_shape, self.nb_classes)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(X, y_onehot, batch_size=self.batch_size, epochs=self.nb_epochs,
                                      verbose=self.verbose, callbacks=self.callbacks)

        self.save_trained_model()
        self.is_fitted = True

        return self
