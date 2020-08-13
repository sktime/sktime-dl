#! /usr/bin/env python
"""Install script for sktime-dl"""

import codecs
import os
import platform
import re
import sys

from pkg_resources import Requirement
from pkg_resources import working_set
from setuptools import find_packages
from setuptools import setup

# raise early warning for incompatible Python versions
if sys.version_info < (3, 6) or sys.version_info >= (3, 8):
    raise RuntimeError(
        "sktime-dl requires Python 3.6 or 3.7 (only with tensorflow>=1.13.1). "
        "The current Python version is %s installed in %s."
        % (platform.python_version(), sys.executable))

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(HERE, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")


def find_install_requires():
    """Return a list of dependencies and non-pypi dependency links.

    A supported version of tensorflow and/or tensorflow-gpu is required. If not
    found, then tensorflow is added to the install_requires list.

    Depending on the version of tensorflow found or installed, either
    keras-contrib or tensorflow-addons needs to be installed as well.
    """

    install_requires = [
        'sktime>=0.4.1',
        'h5py>=2.8.0',
    ]

    # tensorflow version requirements
    # by default, make sure anything already installed is above 1.8.0,
    # or if installing from new get the most recent stable (i.e. not
    # nightly) version
    MINIMUM_TF_VERSION = '1.9.0'
    tf_requires = 'tensorflow>=' + MINIMUM_TF_VERSION

    has_tf_gpu = False
    has_tf = False
    tf = working_set.find(Requirement.parse('tensorflow'))
    tf_gpu = working_set.find(Requirement.parse('tensorflow-gpu'))

    if tf is not None:
        has_tf = True
        tf_version = tf._version

    if tf_gpu is not None:
        has_tf_gpu = True
        tf_gpu_version = tf_gpu._version

    if has_tf_gpu and not has_tf:
        # have -gpu only (1.x), make sure it's above 1.9.0
        # Specify tensorflow-gpu version if it is already installed.
        tf_requires = 'tensorflow-gpu>=' + MINIMUM_TF_VERSION

    install_requires.append(tf_requires)

    # tensorflow itself handled, now find out what add-on package to use
    if (not has_tf and not has_tf_gpu) or (has_tf and tf_version >= '2.1.0'):
        # tensorflow will be up-to-date enough to use most recent
        # tensorflow-addons, the replacement for keras-contrib
        install_requires.append('tensorflow-addons')
    else:
        # fall back to keras-contrib, not on pypi so need to install it
        # separately not printing. TODO
        print(
            'Existing version of tensorflow older than version 2.1.0 '
            'detected. You shall need to install keras-contrib (for tf.keras) '
            'in order to use all the features of sktime-dl. '
            'See https://github.com/keras-team/keras-contrib#install-keras_contrib-for-tensorflowkeras')

    return install_requires


DISTNAME = 'sktime-dl'  # package name is sktime-dl, to have a valid module path, module name is sktime_dl
DESCRIPTION = 'Deep learning extension package for sktime, a scikit-learn ' \
              'compatible toolbox for learning with time series data'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'F. Király'
MAINTAINER_EMAIL = 'f.kiraly@ucl.ac.uk'
URL = 'https://github.com/sktime/sktime-dl'
LICENSE = 'BSD-3-Clause'
DOWNLOAD_URL = 'https://pypi.org/project/sktime-dl/#files'
PROJECT_URLS = {
    'Issue Tracker': 'https://github.com/sktime/sktime-dl/issues',
    'Documentation': 'https://sktime.github.io/sktime-dl/',
    'Source Code': 'https://github.com/sktime/sktime-dl'
}
VERSION = find_version('sktime_dl', '__init__.py')
INSTALL_REQUIRES = find_install_requires()
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']

EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'
        'flaky'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      include_package_data=True,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      )
