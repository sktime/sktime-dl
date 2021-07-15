# -*- coding: utf-8 -*-
"""Time Convolutional Neural Network (CNN) (minus the final output layer)."""

__author__ = "Jack Russon"

from tensorflow import keras
import tensorflow as tf
import numpy as np
import math
from keras_self_attention import   SeqSelfAttention

from sktime_dl.networks._network import BaseDeepNetwork


class TapNetNetwork(BaseDeepNetwork):
    """
    """

    def __init__(
            self,
            dropout=0.5,
            filter_sizes=[256, 256, 128],
            kernel_size=[8, 5, 3],
            dilation=1,
            layers=[500, 300],
            use_rp=True,
            rp_params=[-1, 3],
            use_att=True,
            use_ss=False,
            use_metric=False,
            use_muse=False,
            use_lstm=True,
            use_cnn=True,
            random_state=1,
            padding='same'
    ):
        """
        :param kernel_size: int, specifying the length of the 1D convolution
         window
        :param avg_pool_size: int, size of the average pooling windows
        :param nb_conv_layers: int, the number of convolutional plus average
         pooling layers
        :param filter_sizes: int, array of shape = (nb_conv_layers)
        :param random_state: int, seed to any needed random actions
        """
        super(TapNetNetwork,self).__init__()
        self.random_state = random_state
        self.kernel_size = kernel_size
        self.layers=layers
        self.rp_params=rp_params
        self.filter_sizes = filter_sizes
        self.use_att=use_att
        self.use_ss=use_ss
        self.dilation=dilation
        self.padding=padding

        self.dropout = dropout
        self.use_metric = use_metric
        self.use_muse = use_muse
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn



        # parameters for random projection
        self.use_rp = use_rp
        self.rp_params = rp_params

    def output_conv_size(self,in_size, kernel_size, strides, padding):
        #padding removed for now

        output = int((in_size - kernel_size) / strides) + 1

        return output

    def euclidean_dist(self,x, y):
        # x: N x D
        # y: M x D
        print(x.shape,y.shape)
        n = tf.shape(x)[0]
        m = tf.shape(y)[0]
        d = tf.shape(x)[1]
        #assert d == tf.shape(y)[1]
        x=tf.expand_dims(x,1)
        y=tf.expand_dims(y,0)
        print('xshape,yshape')
        print(x.shape, y.shape)
        x=tf.broadcast_to(x, shape=(n,m,d))
        y=tf.broadcast_to(y,shape=(n,m,d))

        print('xshape,yshape')
        print(x.shape,y.shape)
        print(tf.math.pow(x - y, 2).shape)
        print(tf.math.reduce_sum(tf.math.pow(x - y, 2),axis=2).shape)
        return tf.math.reduce_sum(tf.math.pow(x - y, 2),axis=2)

    def build_network(self, input_shape, nclass,X, y, **kwargs):
        """
        Construct a network and return its input and output layers

        Arguments
        ---------
        input_shape : tuple
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        input_layer=keras.layers.Input(input_shape)

        if self.rp_params[0]< 0:
            dim = input_shape[0]
            self.rp_params = [3, math.floor(dim * 2 / 3)]
        self.rp_group, self.rp_dim = self.rp_params

        if self.use_lstm:

            self.lstm_dim = 128

            x_lstm = keras.layers.LSTM(self.lstm_dim,return_sequences=True)(input_layer)
            print('x_lstm shape')
            print(x_lstm.shape)

            if self.use_att:
                x_lstm = SeqSelfAttention(128, attention_type='multiplicative')(x_lstm)
            print(x_lstm.shape)
            x_lstm = keras.layers.GlobalAveragePooling1D()(x_lstm)

        if self.use_cnn:
            # Covolutional Network
            # input ts: # N * C * L
            if self.use_rp:
                self.conv_1_models = keras.Sequential()

                print('rp_group' + str(self.rp_group))
                for i in range(self.rp_group):

                    self.idx=(np.random.permutation(input_shape[1])[0: self.rp_dim])
                    channel =keras.layers.Lambda(lambda x: tf.gather(x,indices=self.idx,axis=2))(input_layer)
                    # x_conv = x
                    #x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
                    x_conv = keras.layers.Conv1D(self.filter_sizes[0], kernel_size=self.kernel_size[0],
                                                 dilation_rate=self.dilation, strides=1,
                                                 padding=self.padding)(channel)  # N * C * L

                    x_conv = keras.layers.BatchNormalization()(x_conv)
                    x_conv = keras.layers.LeakyReLU()(x_conv)

                    x_conv = keras.layers.Conv1D(self.filter_sizes[1], kernel_size=self.kernel_size[0],
                                                 dilation_rate=self.dilation, strides=1,
                                                 padding=self.padding)(x_conv)
                    x_conv = keras.layers.BatchNormalization()(x_conv)
                    x_conv = keras.layers.LeakyReLU()(x_conv)

                    x_conv = keras.layers.Conv1D(self.filter_sizes[2], kernel_size=self.kernel_size[0],
                                                 dilation_rate=self.dilation, strides=1,
                                                 padding=self.padding)(x_conv)
                    x_conv = keras.layers.BatchNormalization()(x_conv)
                    x_conv = keras.layers.LeakyReLU()(x_conv)
                    print('cnn_shape')
                    print(x_conv.shape)
                    if self.use_att:
                        x_conv = SeqSelfAttention(128, attention_type='multiplicative')(x_conv)
                    print(x_conv.shape)

                    x_conv=keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x_conv)

                    if i == 0:

                        x_conv_sum = x_conv
                    else:
                        x_conv_sum = keras.layers.Concatenate()([x_conv_sum, x_conv])



                x_conv = x_conv_sum


            else:

                x_conv = keras.layers.Conv1D(self.filter_sizes[0], kernel_size=self.kernel_size[0], dilation_rate=self.dilation, strides=1,
                                        padding=self.padding)(input_layer)  # N * C * L

                x_conv = keras.layers.BatchNormalization()(x_conv)
                x_conv = keras.layers.LeakyReLU()(x_conv)

                x_conv = keras.layers.Conv1D(self.filter_sizes[1], kernel_size=self.kernel_size[0], dilation_rate=self.dilation, strides=1,
                                        padding=self.padding)(x_conv)
                x_conv = keras.layers.BatchNormalization()(x_conv)
                x_conv = keras.layers.LeakyReLU()(x_conv)

                x_conv = keras.layers.Conv1D(self.filter_sizes[2], kernel_size=self.kernel_size[0], dilation_rate=self.dilation, strides=1,
                                        padding=self.padding)(x_conv)
                x_conv = keras.layers.BatchNormalization()(x_conv)
                x_conv = keras.layers.LeakyReLU()(x_conv)
                if self.use_att:
                    x_conv=SeqSelfAttention(128)(x_conv)

                x_conv = keras.layers.GlobalAveragePooling1D(data_format='channels_last')(x_conv)

        if self.use_lstm and self.use_cnn:
                x = keras.layers.Concatenate()([x_conv, x_lstm])
        elif self.use_lstm:
            x = x_lstm
        elif self.use_cnn:
            x = x_conv

        #
        # if self.use_att:
        #     print(x.shape)
        #     x = tf.expand_dims(x, 1)
        #     x_proto = SeqSelfAttention(units=att_dim, attention_type='multiplicative')(x)
        #     print(x_proto.shape)
        #
        #     x = self.euclidean_dist(x, x_proto)
        #Mapping section
        x=keras.layers.Dense(self.layers[0], name="fc_")(x)
        x=keras.layers.LeakyReLU(name="relu_")(x)
        x=keras.layers.BatchNormalization(name="bn_")(x)

        x=keras.layers.Dense(self.layers[1],name="fc_2")(x)





        # else:  # if do not use attention, simply use the mean of training samples with the same labels.
        #     print('idx then xshape')
        #     print(idx[0],idx[0].shape)
        #     print(x,x.shape)
        #     self.proto_indices=idx[0]
        #     class_repr=tf.gather(x,idx[0])
        #     print(class_repr.shape)
        #     class_repr = tf.reduce_mean(class_repr,axis=0) # L * 1
        #     print('class_repr', class_repr.shape)
        #
        # proto_list.append(tf.reshape(class_repr,shape=(1,-1)))





       # x=keras.layers.Dense(4,activation='softmax')(x)
        return input_layer, x


