.. _developer-guide:

Developer Guide
===============

This page explains the inner workings of the library and how one can contribute
to its development.


Project Organization
--------------------

The project comprises the following subdirectories:

- ``demo``: Python scripts that illustrate the usage of the library.
- ``docs``: Documentation file written with ``Sphinx``.
- ``diurnal``: Library source code.
- ``test``: Automated test framework written with ``pytest``.


Dependencies
------------

The library is written in Python and relies mainly on the following external
libraries:

- ``Numpy`` (data manipulation)
- ``PyTorch`` (neural networks)
- ``matplotlib`` (data visualization)
- ``scikit-learn`` (evaluation methods)
- ``pytest`` (only used by the test suite, not the actual library)
- ``sphinx`` (documentation only, not for the actual library)

The complete list of dependencies is comprised in the file
``requirements.txt``.


How to Contribute
-----------------

Contributions from any individual are welcome.

Before opening a pull request, please address the following points:

- Document newly added features in the ``docs/`` directory.
- Add appropriate test cases in the ``test/`` directory.
- Ensure that the new feature interoperates well with existing code and follows
  the PEP 8 coding style.

You can execute the Python script ``pre_commit.py`` to ensure that (1) the
source code does follow the PEP 8 style and (2) the test suite passes.

Provided that the points described above are taken into consideration, one can
submit a contribution by opening a pull request on the
`homepage <https://github.com/Vincent-Therrien/diurnal>`_ of the project.
Please provide a rationale for opening the pull request.

Development History
-------------------


