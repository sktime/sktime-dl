#!/bin/bash

# adapted from https://github.com/scikit-learn/scikit-learn/blob/master/build_tools/travis/install.sh

set -e

echo "Setting up conda env ..."
echo "Python version: " "$PYTHON_VERSION"
echo "TF version: " "$TF_VERSION"

# Deactivate the any previously set virtual environment and setup a
# conda-based environment instead
deactivate || :

# Configure conda
conda config --set always_yes true
conda update --quiet conda

# Set up test environment
conda create --name testenv python="$PYTHON_VERSION"

# Activate environment
source activate testenv

# Install tensorflow via pip, tests fail when installed via conda
pip install tensorflow=="$TF_VERSION"

# Install requirements from inside conda environment
pip install cython  # only needed until we provide sktime wheels
pip install -r "$REQUIREMENTS"

# Build sktime-dl
# invokes build_ext -i to compile files
# builds universal wheel, as specified in setup.cfg
python setup.py bdist_wheel

# Install from built wheels
pip install --pre --no-index --no-deps --find-links dist/ sktime-dl

# now need to install keras-contrib for tf.keras instead of standalone keras
# not needed for the tf_version 2.1 env, but does not hurt either. investigate
# conditional installation
echo "Installing keras-contrib ..."
git clone https://www.github.com/keras-team/keras-contrib.git
cd keras-contrib/
python convert_to_tf_keras.py
USE_TF_KERAS=1 python setup.py install
cd ..

set +e
