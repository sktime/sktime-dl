.. image:: https://travis-ci.com/sktime/sktime-dl.svg?branch=master
    :target: https://travis-ci.com/sktime/sktime-dl
.. image:: https://badge.fury.io/py/sktime-dl.svg
    :target: https://badge.fury.io/py/sktime-dl
.. image:: https://badges.gitter.im/sktime/community.svg
    :target: https://gitter.im/sktime/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge


sktime-dl
=========
An extension package for deep learning with Keras for `sktime <https://github.com/alan-turing-institute/sktime>`__, a `scikit-learn <https://github.com/scikit-learn/scikit-learn>`__ compatible Python toolbox for learning with time series and panel data. 

sktime-dl is under development and we welcome new contributors.

Installation
------------

sktime-dl requires `keras-contrib <https://github.com/keras-team/keras-contrib>`__, which is not available on pypi. 

The simplest installation method is:
::

	pip install sktime-dl
	pip install git+https://www.github.com/keras-team/keras-contrib.git
	
sktime-dl is under development. To guarantee that you're using the most up to date code, you can install the development version: 
::
	git clone https://github.com/uea-machine-learning/sktime-dl.git
	cd sktime-dl
	git checkout dev
	git pull origin dev
	pip install . 
	
	pip install git+https://www.github.com/keras-team/keras-contrib.git
	
When installing sktime-dl, `Tensorflow <https://www.tensorflow.org/install/>`__ 1.x will be installed as the backend for Keras. Tensorflow 2.x is currently unsupported. Other backends should be usable in principle but are untested.
	
With these instructions, the networks can be run on your CPU. If you wish to run the networks on an NVIDIA® GPU, extra drivers and toolkits (GPU drivers, CUDA Toolkit, and CUDNN library) need to be installed separately to sktime-dl. See `this page<https://www.tensorflow.org/install/gpu>`__ for more information.

Lastly, if you have a tensorflow version less than 1.15, `tensorflow-gpu needs to be installed<https://www.tensorflow.org/install/gpu>`__ in addition to (or in place of) the tensorflow (no suffix) that will be installed automatically, e.g.:
::
	pip install tensorflow-gpu==1.14

For windows users looking to setup Keras for GPU usage in general for the first time, we can also recommend following `this (unaffiliated) guide<https://github.com/antoniosehk/keras-tensorflow-windows-installation>`__.
	
This package uses the base sktime as a dependency. You can follow the `original instructions <https://alan-turing-institute.github.io/sktime/installation.html>`__ to install this separately or as the development version if wanted. The sktime-dl package currently has API calls up to date with **sktime version 0.3.1**. Updates to sktime may precede sktime-dl updates by some lag time.
	

Overview
--------

A repository for off-the-shelf networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The aim is to define Keras networks able to be directly used within sktime and its pipelining and strategy tools, and by extension scikit-learn, for use in applications and research. Over time, we wish to interface or implement a wide range of networks from the literature in the context of time series analysis.

Currently, we interface with a number of networks for time series classification in particular. A large part of the current toolset serves as an interface to `dl-4-tsc <https://github.com/hfawaz/dl-4-tsc>`__, and implements the following network architectures: 

- Time convolutional neural network (CNN)
- Encoder (Encoder)
- Fully convolutional neural network (FCNN)
- Multi channel deep convolutional neural network (MCDCNN)
- Multi-scale convolutional neural network (MCNN)
- Multi layer perceptron (MLP)
- Residual network (resnet)
- Time Le-Net (tlenet)
- Time warping invariant echo state network (twiesn)

We also interface with InceptionTime, as of writing the strongest deep learning approach to general time series classification.

- Inception network, singular. 

Meta-functionality
~~~~~~~~~~~~~~~~~~

-	Hyper-parameter tuning (through calls to sci-kit learns Grid and RandomizedSearch tools, currently) 
-	Ensembling methods (over different random initialisations for stability) 
These act as wrappers to networks, and can be used in high-level and experimental pipelines as with any sktime model. 

Documentation
-------------

sktime-dl is an extension package to sktime, primarily introducing different learning algorithms. All `examples <https://github.com/alan-turing-institute/sktime/tree/master/examples>`__ and `documentation <https://alan-turing-institute.github.io/sktime/>`__ from the base sktime apply to this package. 

Documentation specifically for sktime-dl shall be produced in due course.

Contributors
------------
Former and current active contributors are as follows.

James Large (@James-Large, `@jammylarge <https://twitter.com/jammylarge>`__, james.large@uea.ac.uk), Aaron Bostrom (@ABostrom), Hassan Ismail Fawaz (@hfawaz), Markus Löning (@mloning)
