.. -*- mode: rst -*-

|Azure|_ |Travis|_ |Codecov|_ |CircleCI|_ |Nightly wheels|_ |PythonVersion|_ |PyPi|_ |DOI|_

.. |Azure| image:: https://dev.azure.com/scikit-learn/scikit-learn/_apis/build/status/scikit-learn.scikit-learn?branchName=main
.. _Azure: https://dev.azure.com/scikit-learn/scikit-learn/_build/latest?definitionId=1&branchName=main

.. |Travis| image:: https://api.travis-ci.com/scikit-learn/scikit-learn.svg?branch=main
.. _Travis: https://travis-ci.com/scikit-learn/scikit-learn

.. |Codecov| image:: https://codecov.io/github/scikit-learn/scikit-learn/badge.svg?branch=main&service=github
.. _Codecov: https://codecov.io/github/scikit-learn/scikit-learn?branch=main

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn/scikit-learn/tree/main.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn/scikit-learn

.. |Nightly wheels| image:: https://github.com/scikit-learn/scikit-learn/workflows/Wheel%20builder/badge.svg?event=schedule
.. _`Nightly wheels`: https://github.com/scikit-learn/scikit-learn/actions?query=workflow%3A%22Wheel+builder%22+event%3Aschedule

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue
.. _PythonVersion: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue

.. |PyPi| image:: https://badge.fury.io/py/scikit-learn.svg
.. _PyPi: https://badge.fury.io/py/scikit-learn

.. |DOI| image:: https://zenodo.org/badge/21369/scikit-learn/scikit-learn.svg
.. _DOI: https://zenodo.org/badge/latestdoi/21369/scikit-learn/scikit-learn

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


![Alt text](/sklearn/docs/Images/Sklearn1.png?raw=true "Optional Title")

**sq-learn** is a Python framework integrating classical machine learning algorithms in the tightly-knit world of scientific Python packages (numpy, scipy, matplotlib) with the quantum world.

It aims to provide simple and efficient solutions to learning problems that are accessible to everybody and reusable in various contexts: machine-learning as a versatile tool for science and engineering.

It can be used to run either classical experiments or simulation of the quantum counterpart.

Up to now in this framework only PCA and K-means algorithm can be simulated in their quantum counterpart.

=======

In this repository, you can find the experiments and numerical simulations done for quantum pure state tomography, phase estimation, consistent phase estimation, and amplitude estimation.

Furthermore, we also include our experiments for what concerns the application of our sq-learn framework to cybersecurity problems.
