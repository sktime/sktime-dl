__author__ = "Mahak Kothari"

from sktime_dl.deeplearning.base.estimators import BaseDeepClassifier
from sktime_dl.deeplearning.lstm._base import LSTMNetwork
from sktime_dl.utils._data import check_and_clean_data, \
    check_and_clean_validation_data
from sklearn.utils import check_random_state
from tensorflow import keras
import numpy as np


class LSTMClassifier(BaseDeepClassifier, LSTMNetwork):
    """ Multivariate Time series classification using Long Short Term Memory (LSTM).    
    (Many to One classification)

    :param nb_epochs: int, the number of epochs to train the model
    :param batch_size: int, the number of samples per gradient update.    
    :param units: list, Number of hidden units in each layer.    
    :param random_state: int, or sklearn Random.state
    :param verbose: boolean, whether to output extra information
    :param model_name: string, the name of this model for printing and
    file writing purposes
    :param model_save_directory: string, if not None; location to save
    the trained keras model in hdf5 format
    :param callbacks: list of tf.keras.callbacks.Callback objects
    """

    def __init__(
            self,
            nb_epochs=200,
            batch_size=16,            
            units=[64, 128],            
            random_state=0,
            verbose=False,
            model_name="Lstm_classifier",
            model_save_directory=None,
            callbacks = None
    ):
        super(LSTMClassifier, self).__init__(
            model_save_directory=model_save_directory,
            model_name=model_name)
                
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.units = units
        self.random_state = random_state
        self.verbose = verbose
        self._is_fitted = False
        self.callbacks = None

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
        ----------
        output : a compiled Keras Model
        """
        input_layer, output_layer = self.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(
            units=nb_classes, activation="softmax"
        )(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        return model

    def Rearrange_data(self, X):
        """
        Rearrange the data into the following format (Trials, Timesteps, features)
        ----------
        X: The data which needs to be rearranged.

        Returns
        ----------
        output: The rearranged data.
        """
        X_final = []

        for row in range(len(X)):
            row_array = X.iloc[row, : ].values
            row_vector = row_array.reshape(-1, 1)
            row_matrix = []
            for col in range(len(row_vector)):
                column = row_vector[col][0].values        
                row_matrix.append(np.array(column))
            X_final.append(np.array(row_matrix))        

        return(np.array(X_final))

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
        self.random_state = check_random_state(self.random_state)

        if self.callbacks is None:
            self.callbacks = []

        X = check_and_clean_data(X, y, input_checks=input_checks)        
                    
        X = self.Rearrange_data(X)
          
        y_onehot = self.convert_y(y)

        validation_X, validation_y = check_and_clean_validation_data(validation_X, validation_y,
                                            self.label_encoder,
                                            self.onehot_encoder)
        validation_X = self.Rearrange_data(validation_X)
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
            validation_data=(validation_X, validation_y)
        )

        self._is_fitted = True
        self.save_trained_model()

        return self
