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


Development Objectives
----------------------

The ``diurnal`` library aims at facilitating research in the field of
molecule structure prediction and analysis. As such, its development emphasizes
the following points:

- **Traceability**: Users should always know the source of data and how to use
  them. For instance, when downloading datasets, the
  library produces an informative file that lists when the file was obtain,
  from which URL, and how the data are represented. This avoids letting users
  figure that out by themselves.
- **Inspectability**: The source code should be as easy to understand as
  possible to let reviewers confirm the validity of the results produced by
  the library. *Readability is favored over performances. Prefer functions that
  are easy to understand but slow over highly optimized, hard-to-understand
  functions*.
- **Modularity**: The modules of the library should be easy to use and
  understand in isolation. Limit interactions between modules.


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

The complete list of dependencies for the library is comprised in the file
``requirements.txt``. The file ``docs/requirements.txt`` lists the dependencies
required to build the documentation. The file ``test/requirements.txt`` lists
the dependencies required to execute the test framework.


Test Suite
----------

The project comprises a
`test suite <https://github.com/Vincent-Therrien/diurnal/tree/main/test>`_ that
validates the features of the library. It is written with the
`pytest <https://docs.pytest.org/en/7.4.x/>`_ library. Each feature of the
library should be validated with a test case.


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
Please provide a rationale for opening the pull request and references, if
relevant.

Development History
-------------------


