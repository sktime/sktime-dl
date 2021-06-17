NOTE: sktime-dl is currenlty being updated to work correctly with sktime 0.6, and wwill be fully relaunched over the summer. The plan is

1. Update it to be compliant with sktime 0.6 (currently works with sktime 0.4)
2. Update classifiers (documentation etc)
3. Import pytorch, add a pytorch classifier
4. Add a forecasting module
5. Review literature on the latest dl classifiers, assimilate and evaluate any worth including
6. Update devops so it exactly mirror sktime


|travis|_ |pypi|_ |gitter|_ |Binder|_

.. |travis| image:: https://img.shields.io/travis/com/sktime/sktime-dl/master?logo=travis
.. _travis: https://travis-ci.com/sktime/sktime-dl

.. |pypi| image:: https://img.shields.io/pypi/v/sktime-dl
.. _pypi: https://pypi.org/project/sktime-dl/

.. |gitter| image:: https://img.shields.io/gitter/room/alan-turing-institute/sktime?logo=gitter
.. _gitter: https://gitter.im/sktime/community

.. |binder| image:: https://mybinder.org/badge_logo.svg
.. _Binder: https://mybinder.org/v2/gh/sktime/sktime-dl/master?filepath=examples

sktime-dl
=========

An extension package for deep learning with Tensorflow/Keras for `sktime <https://github.com/alan-turing-institute/sktime>`__, a `scikit-learn <https://github.com/scikit-learn/scikit-learn>`__ compatible Python toolbox for learning with time series and panel data. 

sktime-dl is under development and currently in a state of flux. 


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
- Residual network (ResNet)
- Time Le-Net (TLeNet)
- Time warping invariant echo state network (TWIESN)

We also interface with `InceptionTime <https://github.com/hfawaz/InceptionTime>`__, as of writing the strongest deep learning approach to general time series classification. 

- Inception network, singular. 

Regression
~~~~~~~~~~

Most of the classifier architectures have been adapted to also provide regressors. These are:

- Time convolutional neural network (CNN)
- Encoder (Encoder)
- Fully convolutional neural network (FCNN)
- Multi layer perceptron (MLP)
- Residual network (ResNet)
- Time Le-Net (TLeNet)
- InceptionTime (Inception)

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

Example notebooks for sktime-dl usage can be found under the examples folder.

Contributors
------------
Former and current active contributors are as follows:

James Large (@James-Large, `@jammylarge <https://twitter.com/jammylarge>`__, james.large@uea.ac.uk), Aaron Bostrom (@ABostrom), Hassan Ismail Fawaz (@hfawaz), Markus Löning (@mloning), @Withington
