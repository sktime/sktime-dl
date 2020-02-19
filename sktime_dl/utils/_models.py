# Model utility functions

__author__ = "Withington"

from pathlib import Path


def save_trained_model(model, model_save_directory, model_name, save_format='h5'):
    """
    Saves the model to an HDF file.
    
    Saved models can be reinstantiated via `keras.models.load_model`.
    Parameters
    ----------
    save_format: string
        'h5'. Defaults to 'h5' currently but future releases
        will default to 'tf', the TensorFlow SavedModel format.
    """
    if save_format is not 'h5':
        raise ValueError("save_format must be 'h5'. This is the only format currently supported.")
    if model_save_directory is not None:
        if model_name is None:
            file_name = 'trained_model.hdf5'
        else:
            file_name = model_name + '.hdf5'
        path = Path(model_save_directory) / file_name
        model.save(path) # Add save_format here upon migration from keras to tf.keras
