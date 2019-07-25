.. image:: https://travis-ci.com/uea-machine-learning/sktime-dl.svg?branch=master
    :target: https://travis-ci.com/uea-machine-learning/sktime-dl

sktime-dl
=========
An additional deep learning with Keras toolset for `sktime <https://github.com/alan-turing-institute/sktime>`__, a `scikit-learn <https://github.com/scikit-learn/scikit-learn>`__ compatible Python toolbox for learning with time series and panel data. 

The package is under active development. Currently, classification models based off the the networks in `dl-4-tsc <https://github.com/hfawaz/dl-4-tsc>`__ have been implemented, as well as an example of a tuned network for future development. 

Installation
------------
This package uses the base sktime as a dependency. Follow the `original instructions <https://help.github.com/en/articles/changing-a-remotes-url>`__ to install this. 

For the deep-learning part of sktime-dl, you need:

- `Keras <https://github.com/keras-team/keras>`__
- `keras-contrib <https://github.com/keras-team/keras-contrib>`__ 
- and a compatible backend for Keras, one of 

    - `tensorflow <https://www.tensorflow.org/install/>`__ (confirmed working, v1.8.0)
    - `theano <http://deeplearning.net/software/theano/install.html#install>`__ (untested)
    - `CNTK <https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine>`__ (untested)

If you want to run the networks on a GPU, `CUDNN <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/>`__ is also required to be able to utilise your GPU. 

For windows users, we recommend following `this <https://github.com/antoniosehk/keras-tensorflow-windows-installation>`__ (unaffiliated) guide.

For linux users, all of these points should hopefully be relatively straight forward via simple pip-commands and conversions from the previous link.

For mac users, I am unfortunately unsure of the best processes for installing these. If you have links to a tested and up-to-date guide, let us know (@James-Large).

Overview
--------

A repository for off-the-shelf networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The aim is to define Keras networks able to be directly used within sktime and its pipelining and strategy tools, and by extension scikit-learn, for use in applications and research. Overtime, we wish to interface or reimplement networks from the literature in the context of time series analysis.

Currently, we interface with a number of networks for time series classification in particular. 

dl-4-tsc interfacing
~~~~~~~~~~~~~~~~~~~
This toolset currently serves as an interface to `dl-4-tsc <https://github.com/hfawaz/dl-4-tsc>`__, and implements the following network archtiectures: 

- Time convolutional neural network (CNN)
- Encoder (Encoder)
- Fully convolutional neural network (FCNN)
- Multi channel deep convolutional neural network (MCDCNN)
- Multi-scale convolutional neural network (MCNN)
- Multi layer perceptron (MLP)
- Residual network (resnet)
- Time Le-Net (tlenet)
- Time warping invariant echo state network (twiesn)


Documentation
-------------
The full API documentation to the base sktime and an introduction can be found `here <https://alan-turing-institute.github.io/sktime/>`__.
Tutorial notebooks for currently stable functionality are in the `examples <https://github.com/alan-turing-institute/sktime/tree/master/examples>`__ folder.

Documentation for sktime-dl shall be produced in due course.

Contributors
------------
Former and current active contributors are as follows.

sktime-dl
~~~~~~~~~

James Large (@James-Large), Aaron Bostrom (@ABostrom), Hassan Ismail Fawaz (@hfawaz), Markus Löning (@mloning)

sktime
~~~~~~

Project management: Jason Lines (@jasonlines), Franz Király (@fkiraly)

Design: Anthony Bagnall(@TonyBagnall), Sajaysurya Ganesh (@sajaysurya), Jason Lines (@jasonlines), Viktor Kazakov (@viktorkaz), Franz Király (@fkiraly), Markus Löning (@mloning)

Coding: Sajaysurya Ganesh (@sajaysurya), Bagnall(@TonyBagnall), Jason Lines (@jasonlines), George Oastler (@goastler), Viktor Kazakov (@viktorkaz), Markus Löning (@mloning)

We are actively looking for contributors. Please contact @fkiraly or @jasonlines for volunteering or information on paid opportunities, or simply raise an issue in the tracker.
