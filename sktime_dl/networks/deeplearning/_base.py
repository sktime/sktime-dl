# Base class for networks

__author__ = "Withington"


class BaseDeepNetwork():

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
        raise NotImplementedError('this is an abstract method')
    