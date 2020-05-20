__author__ = "James Large"

from tensorflow import keras
import numpy as np

from sktime_dl.deeplearning.base.estimators import BaseDeepClassifier
from sktime_dl.deeplearning.mlstmfcn._base import MLSTMFCNNetwork
from sktime_dl.utils import check_and_clean_data


class MLSTMFCNClassifier(BaseDeepClassifier, MLSTMFCNNetwork):
    """
    Multivariate Long Short Term Memory Fully Convolutional Network(MLSTMFCN)

    Adapted from the implementation by Karim et. al

    https://github.com/houshd/MLSTM-FCN

    Network originally defined in:

    @article{Karim_2019,
    title={Multivariate LSTM-FCNs for time series classification},
    volume={116},
    ISSN={0893-6080},
    url={http://dx.doi.org/10.1016/j.neunet.2019.04.014},
    DOI={10.1016/j.neunet.2019.04.014},
    journal={Neural Networks},
    publisher={Elsevier BV},
    author={Karim, Fazle and Majumdar, Somshubra and
            Darabi, Houshang and Harford, Samuel},
    year={2019},
    month={Aug},
    pages={237–245}
    }
    """

    def __init__(
            self,
            nb_lstm_cells=64,
            nb_epochs=250,
            batch_size=128,
            callbacks=None,
            random_seed=0,
            verbose=False,
            model_name="MLSTMFCN",
            model_save_directory=None,
    ):
        super().__init__(
            model_name=model_name, model_save_directory=model_save_directory
        )
        MLSTMFCNNetwork.__init__(self, nb_lstm_cells=nb_lstm_cells,
                                 random_seed=random_seed)
        """
        :param nb_epochs: int, the number of epochs to train the model
        :param batch_size: int, specifying the length of the 1D convolution
        window
        :param callbacks: list of tf.keras.callbacks.Callback objects
        :param random_seed: int, seed to any needed random actions
        :param verbose: boolean, whether to output extra information
        :param model_name: string, the name of this model for printing and file
         writing purposes
        :param model_save_directory: string, if not None; location to save the
         trained keras model in hdf5 format
        """

        self.verbose = verbose
        self.is_fitted = False

        # calced in fit
        self.input_shape = None
        self.history = None

        # predefined
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.callbacks = callbacks if callbacks is not None else []

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
        output_layer = keras.layers.Dense(nb_classes, activation="softmax")(
            output_layer
        )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(lr=1e-3),
            metrics=["accuracy"],
        )

        # if user hasn't provided a custom ReduceLROnPlateau via init already,
        # add the default from literature
        if not any(
                isinstance(callback, keras.callbacks.ReduceLROnPlateau)
                for callback in self.callbacks
        ):
            factor = 1. / np.cbrt(2)

            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=factor, patience=100, min_lr=1e-4,
                cooldown=0, optimization_mode='auto'
            )
            self.callbacks.append(reduce_lr)

        return model

    def fit(self, X, y, input_checks=True, **kwargs):
        """
        Build the classifier on the training set (X, y)
        ----------
        X : array-like or sparse matrix of shape = [n_instances, n_columns]
            The training input samples.  If a Pandas data frame is passed,
            column 0 is extracted.
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

        # class weighting used in paper
        # todo likely can heavily simplify this, currently direct conversion
        #  from source code
        recip_freq = len(y) / (len(self.label_encoder.classes_) *
                               np.bincount(
                                   self.label_encoder.transform(y)).astype(
                                   np.float64))
        class_weight = recip_freq[self.label_encoder.transform(np.unique(y))]

        # ignore the number of instances, X.shape[0], just want the shape of
        # each instance
        self.input_shape = X.shape[1:]

        self.batch_size = int(min(X.shape[0] / 10, self.batch_size))

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
            class_weight=class_weight
        )

        self.save_trained_model()
        self.is_fitted = True

        return self

if __name__ == '__main__':
    from sktime_dl.deeplearning.tests.test_classifiers import test_basic_univariate

    test_basic_univariate(MLSTMFCNClassifier(nb_epochs=2))