.. -*- mode: rst -*-

|PythonVersion|_ |PyPi|

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue
.. _PythonVersion: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue

.. |PyPi| image:: https://badge.fury.io/py/scikit-learn.svg
.. _PyPi: https://badge.fury.io/py/scikit-learn


.. |PythonMinVersion| replace:: 3.6
.. |NumPyMinVersion| replace:: 1.13.3
.. |SciPyMinVersion| replace:: 0.19.1
.. |JoblibMinVersion| replace:: 0.11
.. |ThreadpoolctlMinVersion| replace:: 2.0.0
.. |MatplotlibMinVersion| replace:: 2.1.1
.. |Scikit-ImageMinVersion| replace:: 0.13
.. |PandasMinVersion| replace:: 0.25.0
.. |SeabornMinVersion| replace:: 0.9.0
.. |PytestMinVersion| replace:: 5.0.1

.. image:: sklearn/docs/Image/Sklearn1.png
   :alt: Alternative text

**sq-learn** is a Python framework integrating quantum routines with classical machine learning algorithms such that we can simulate their quantum counterpart as
we were using a fault-tolerant quantum computer.

It aims to provide simple and efficient solutions to learning problems that are accessible to everybody and reusable in various contexts: quantum 
machine-learning as a versatile tool for science and engineering.

It can be used to run either classical or quantum machine learning experiments.

Up to now only PCA and K-means algorithm can be simulated in their quantum counterpart.

=======

The framework rests on the QuantumUtility.py module in which all the main quantum routines are implemented.
It is important to be aware that the quantum machine learning algorithms that we use can be consider as randomized approximation algorithm. They are formalized in theorems where the approximation error, the probability of failure, and the running time are reported. The approximation error is expected by the quantum computational
paradigm and not by techonological limits. Using this framework you will simulate QML algorithms by simulating their approximation error and probability of failure.
In this way, you can check how the performance, specific to the problem you are solving, are affected by the approximated solution with respect to the exact one of 
classical machine learning algorithms. 

Moeover, you will be able to compare running time performances between classical and quantum machine learning algorithms since the approximation errors and the probabilty of failure are running time parameters.


