# Base class for networks

__author__ = "Withington"

class BaseDeepNetwork():

    def build_network(self, input_shape, **kwargs):
        """
        Construct an un-compiled, un-trained, keras model
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        Returns
        -------
        output : an un-compiled Keras Model
        """
        raise NotImplementedError('this is an abstract method')
    