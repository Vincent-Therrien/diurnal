Diurnal Test Suite
==================

The directory ``diurnal/test`` contains the test suite of the project. It is
written with the ``pytest`` Python library


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


Test Data
---------

The directory ``diurnal/test/data`` contains files intended to validate the
features of the library. The following list presents the source of the files
contained in each subdirectory.

- ``diurnal/test/data/ct`` contains ``ct`` files that represent the RNA
  structure of molecules of the ``archiveII`` dataset.
