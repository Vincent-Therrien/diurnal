Diurnal Test Suite
==================

The directory ``diurnal/test`` contains the test suite of the project.

The test suite is written with the ``pytest`` Python library. More information
about the library can be found in its official documentation at
https://docs.pytest.org/en/7.1.x/contents.html.


Test Execution
--------------

Run the following command to execute the test suite:

.. code-block:: bash

   cd diurnal/test && pytest


Test Environment
----------------

The system executing the test suite must satisfy the following conditions:

- The library ``diurnal`` and its dependencies must be installed and accessible
  by the Python interpreter.
- The interpreter executing the test suite must have read and write access to
  the directory ``diurnal/test``.


Test Cases
----------

The directory comprises the following sets of test cases:

- ``test_database.py`` validates the module ``diurnal.database``.
- ``test_evaluate.py`` validates the module ``diurnal.evaluate``.
- ``test_structure.py`` validates the module ``diurnal.structure``.
- ``test_train.py`` validates the module ``diurnal.train``.


Test Data
---------

The directory ``diurnal/test/data`` contains files intended to validate the
features of the library. The following list presents the source of the files
contained in each subdirectory.

- ``diurnal/test/data/ct`` contains ``ct`` files that represent the RNA
  structure of molecules of the ``archiveII`` dataset.
