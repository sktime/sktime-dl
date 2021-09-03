__author__ = "Jack Russon"

from tensorflow import keras
from keras_self_attention import SeqSelfAttention
import pandas as pd
import numpy as np
from sktime_dl.networks._network import BaseDeepNetwork


class CNTCNetwork(BaseDeepNetwork):
    """Combining contextual neural networks for time series classification

        Adapted from the implementation from Fullah et. al

       https://github.com/AmaduFullah/CNTC_MODEL/blob/master/cntc.ipynb

        Network originally defined in:

        @article{FULLAHKAMARA202057,
        title = {Combining contextual neural networks for time series classification},
        journal = {Neurocomputing},
        volume = {384},
        pages = {57-66},
        year = {2020},
        issn = {0925-2312},
        doi = {https://doi.org/10.1016/j.neucom.2019.10.113},
        url = {https://www.sciencedirect.com/science/article/pii/S0925231219316364},
        author = {Amadu {Fullah Kamara} and Enhong Chen and Qi Liu and Zhen Pan},
        keywords = {Time series classification, Contextual convolutional neural networks, Contextual long short-term memory, Attention, Multilayer perceptron},
       }
        """
    def __init__(
            self,
            random_state=0,
            rnn_layer=64,
            filter_sizes=[16,8],
            kernel_sizes=[1,1],
            lstm_size=8,
            dense_size=64


    ):
        """
        :param random_state: int, seed to any needed random actions
        :param rnn_layer: int, filter size for rnn layer
        :param filter_sizes: int, array of shape 2, filter sizes for two convolutional layers
        :param kernel_sizes: int,array of shape 2,  kernel size for two convolutional layers
        :param lstm_size: int, filter size of lstm layer
        :param dense_size: int, size of dense layer
        """

        self.random_state = random_state
        self.rnn_layer = rnn_layer
        self.filter_sizes=filter_sizes
        self.kernel_sizes=kernel_sizes
        self.lstm_size=lstm_size
        self.dense_size=dense_size


    def build_network(self, input_shape, **kwargs):
        """
        Construct a network and return its input and output layers
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        input_layers=[]
        # This is the CCNN arm
        X2 = keras.layers.Input(input_shape)
        self.dropout=0.2
        rnn1 = keras.layers.SimpleRNN(self.rnn_layer*input_shape[1], activation='relu', use_bias=True, kernel_initializer='glorot_uniform')(X2)
        rnn1 = keras.layers.BatchNormalization()(rnn1)
        rnn1 = keras.layers.Dropout(self.dropout)(rnn1)
        rnn1 = keras.layers.Reshape((64, input_shape[1]))(rnn1)
        X1 = keras.layers.Input(input_shape)
        input_layers.append(X1)
        input_layers.append(X2)
        conv1 = keras.layers.Conv1D(self.filter_sizes[0], self.kernel_sizes[0], activation='relu', kernel_initializer='glorot_uniform')(X1)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Dropout(self.dropout)(conv1)
        conv1 = keras.layers.Dense(input_shape[1], input_shape=(input_shape[0], keras.backend.int_shape(conv1)[2]))(conv1)
        conc1 = keras.layers.Concatenate(axis=-2, name="contextual_convolutional_layer")([conv1, rnn1])
        conv2 = keras.layers.Conv1D(self.filter_sizes[1], self.kernel_sizes[1], activation='relu', kernel_initializer='glorot_uniform',
                       name="standard_cnn_layer")(
            conc1)
        conv2 = keras.layers.Dense(input_shape[1], input_shape=(input_shape[0], keras.backend.int_shape(conv2)[2]))(conv2)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Dropout(0.1)(conv2)


        #CLSTM ARM
        X3 = keras.layers.Input(input_shape)
        input_layers.append(X3)
        lstm11 = keras.layers.LSTM(self.lstm_size*input_shape[1], return_sequences=False, kernel_initializer='glorot_uniform', activation='relu')(X3)
        lstm11 = keras.layers.Reshape((self.lstm_size, input_shape[1]))(lstm11)
        lstm11 = keras.layers.Dropout(self.dropout)(lstm11)
        merge = keras.layers.concatenate([conv2, lstm11], axis=-2)

        avg = keras.layers.MaxPooling1D(pool_size=1, strides=None, padding='valid')(merge)
        avg = keras.layers.Dropout(0.1)(avg)
        att = SeqSelfAttention(attention_width=10,
                             attention_activation='sigmoid',
                              name='Attention',
                              attention_type='multiplicative'
                              )(avg)

        att = keras.layers.Dropout(0.1)(att)
        mlp1 = keras.layers.Dense(self.dense_size, kernel_initializer='glorot_uniform', activation='relu')(att)
        mlp1 = keras.layers.Dropout(0.1)(mlp1)
        mlp2 = keras.layers.Dense(self.dense_size, kernel_initializer='glorot_uniform', activation='relu')(mlp1)
        mlp2 = keras.layers.Dropout(0.1)(mlp2)
        #flat5 = keras.layers.GlobalAveragePooling1D()(att)
        flat5=keras.layers.Flatten()(mlp2)
        return input_layers, flat5

    def prepare_input(self,X):
        """"Prepares input for CLSTM arm of model """
        if X.shape[2]== 1:
            trainX2 = X.reshape([X.shape[0],X.shape[1]])
            pd_trainX = pd.DataFrame(trainX2)
            roll_win1 = pd_trainX.rolling(window=3).mean()  ### contextual feature(P)
            roll_win1 = roll_win1.fillna(0)
            trainX3 = np.concatenate((trainX2, roll_win1), axis=1)
            trainX3 = keras.backend.variable(trainX3)
            trainX3 = keras.layers.Dense(trainX2.shape[1], input_shape=(trainX3.shape[1:]))(trainX3)
            trainX3 = keras.backend.eval(trainX3)
            trainX4 = trainX3.reshape((trainX3.shape[0], trainX3.shape[1], 1))
        else:
            lst=[]
            for i in range(X.shape[2]):
                trainX2 = X[:,:,i]
                pd_trainX = pd.DataFrame(trainX2)
                roll_win1 = pd_trainX.rolling(window=3).mean()  ### contextual feature(P)
                roll_win1 = roll_win1.fillna(0)
                trainX3 = np.concatenate((trainX2, roll_win1), axis=1)
                trainX3 = keras.backend.variable(trainX3)
                trainX3 = keras.layers.Dense(trainX2.shape[1], input_shape=(trainX3.shape[1:]))(trainX3)
                trainX3 = keras.backend.eval(trainX3)
                trainX5 = trainX3.reshape((trainX3.shape[0], trainX3.shape[1], 1))
                lst.append(trainX5)
            trainX4= np.concatenate(lst,axis=2)
        return trainX4