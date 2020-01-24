# Model utility functions

__author__ = "Withington"


def save_trained_model(model, model_save_directory, model_name):
    if model_save_directory is not None:
        if model_name is None:
            model.save(model_save_directory + 'trained_model.hdf5')
        else:
            model.save(model_save_directory + model_name + '.hdf5')
