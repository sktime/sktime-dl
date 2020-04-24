#!/bin/bash

# adapted from https://github.com/scikit-learn/scikit-learn/blob/master/build_tools/travis/install.sh

set -e

echo "$PYTHON_VERSION"
echo "$TF_VERSION"

make_conda() {
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    # If Travvis has language=generic (e.g. for macOS), deactivate does not exist. `|| :` will pass.
    deactivate || :

    # Install miniconda
    echo "Setting up conda env ..."
#    wget https://repo.continuum.io/miniconda/"$MINICONDA_VERSION" -O miniconda.sh
#    MINICONDA=$HOME/miniconda
#    chmod +x miniconda.sh && ./miniconda.sh -b -p "$MINICONDA"
#    export PATH=$MINICONDA/bin:$PATH
    conda config --set always_yes true
    conda update --quiet conda

    # Set up test environment
    conda create --name testenv python="$PYTHON_VERSION" tensorflow="$TF_VERSION"

    # Activate environment
    conda activate testenv

    # Install requirements from inside conda environment
    pip install cython  # only needed until we provide sktime wheels
    pip install -r "$REQUIREMENTS"

    # now need to install keras-contrib for tf.keras instead of standalone keras
    # not needed for the tf_version 2.1 env, but does not hurt either. investigate
    # conditional installation
    echo "Installing keras-contrib ..."
    git clone https://www.github.com/keras-team/keras-contrib.git
    cd keras-contrib
    python convert_to_tf_keras.py
    USE_TF_KERAS=1 python setup.py install
    cd ..
}

# requirements file
make_conda "$REQUIREMENTS"

# Build sktime-dl
# invokes build_ext -i to compile files
# builds universal wheel, as specified in setup.cfg
python setup.py bdist_wheel
ls dist -lh # list built wheels

# Install from built wheels
pip install --pre --no-index --no-deps --find-links dist/ sktime-dl

set +e
