.. image:: https://travis-ci.com/sktime/sktime-dl.svg?branch=master
    :target: https://travis-ci.com/sktime/sktime-dl
.. image:: https://badge.fury.io/py/sktime-dl.svg
    :target: https://badge.fury.io/py/sktime-dl
.. image:: https://badges.gitter.im/sktime/community.svg
    :target: https://gitter.im/sktime/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge


sktime-dl
=========
An extension package for deep learning with Tensorflow/Keras for `sktime <https://github.com/alan-turing-institute/sktime>`__, a `scikit-learn <https://github.com/scikit-learn/scikit-learn>`__ compatible Python toolbox for learning with time series and panel data. 

sktime-dl is under development and we welcome new contributors.

Installation
------------

The simplest installation method is to install in a new environment via pip:
::
	# if using anaconda for environment management
	conda create -n sktime-dl python=3.6
	conda activate sktime-dl
	
	# if using virtualenv for environment management
	virtualenv sktime-dl
	source bin/activate            #unix
	sktime-dl\Scipts\activate      #windows
	
	pip install sktime-dl
	
sktime-dl is under development. To ensure that you're using the most up to date code, you can instead install the development version in your environment: 
::
	git clone https://github.com/sktime-dl/sktime-dl.git
	cd sktime-dl
	git checkout dev
	git pull origin dev
	pip install . 
	
When installing sktime-dl from scratch, the latest stable version of `Tensorflow <https://www.tensorflow.org/install/>`__ 2.x will be installed. Tensorflow 1.x is also supported beyond 1.9, if you have an existing installation in your environment that you wish to maintain. 
	
Users with Tensorflow versions older than 2.1.0 shall also need to install `keras-contrib <https://github.com/keras-team/keras-contrib>`__ after installing sktime-dl, using the `installation instructions for tf.keras <https://github.com/keras-team/keras-contrib#install-keras_contrib-for-tensorflowkeras>`__. 
	
Using GPUS
~~~~~~~~~~
	
With the above instructions, the networks can be run out the box on your CPU. If you wish to run the networks on an NVIDIA® GPU, extra drivers and toolkits (GPU drivers, CUDA Toolkit, and CUDNN library) need to be installed separately to sktime-dl. See `this page <https://www.tensorflow.org/install/gpu#software_requirements>`__ for links and instructions, and also `this page <https://www.tensorflow.org/install/source#tested_build_configurations>`__ for a list of definite versioning compatabilities.       
	
Overview
--------

A repository for off-the-shelf networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The aim is to define Keras networks able to be directly used within sktime and its pipelining and strategy tools, and by extension scikit-learn, for use in applications and research. Over time, we wish to interface or implement a wide range of networks from the literature in the context of time series analysis.

Classification
~~~~~~~~~~~~~~

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

We also interface with `InceptionTime <https://github.com/hfawaz/InceptionTime>`__, as of writing the strongest deep learning approach to general time series classification. 

- Inception network, singular. 

Regression
~~~~~~~~~~

Most of the classifier architectures have been adapted to provide regressors. These are:

- Time convolutional neural network (CNN)
- Encoder (Encoder)
- Fully convolutional neural network (FCNN)
- Multi layer perceptron (MLP)
- Residual network (resnet)
- Time Le-Net (tlenet)
- InceptionTime (inception)

Forecasting
~~~~~~~~~~~

The regression networks can also be used to perform time series forecasting via sktime's `reduction strategies <https://alan-turing-institute.github.io/sktime/examples/forecasting.html#Reduction-strategies>`__. 

We aim to incorporate bespoke forecasting networks in future updates, both specific architectures and general RNNs/LSTMs. 

Meta-functionality
~~~~~~~~~~~~~~~~~~

-	Hyper-parameter tuning (through calls to sci-kit learn's Grid and RandomizedSearch tools, currently) 
-	Ensembling methods (over different random initialisations for stability) 
These act as wrappers to networks, and can be used in high-level and experimental pipelines as with any sktime model. 

Documentation
-------------

sktime-dl is an extension package to sktime, primarily introducing different learning algorithms. All `examples <https://github.com/alan-turing-institute/sktime/tree/master/examples>`__ and `documentation <https://alan-turing-institute.github.io/sktime/>`__ on higher level funtionality and usage from the base sktime apply to this package. 

Documentation specifically for sktime-dl shall be produced in due course.

Contributors
------------
Former and current active contributors are as follows:

James Large (@James-Large, `@jammylarge <https://twitter.com/jammylarge>`__, james.large@uea.ac.uk), Aaron Bostrom (@ABostrom), Hassan Ismail Fawaz (@hfawaz), Markus Löning (@mloning), @Withington
