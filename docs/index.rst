.. diurnal documentation master file, created by
   sphinx-quickstart on Mon Apr 24 21:50:08 2023.

.. Generate the source code documentation file with `sphinx-apidoc -o ./source ../diurnal`

diurnal
=======

``diurnal`` is a Python library that helps predicting RNA secondary structures.

This page presents an `overview <overview>`_ of the project, its
`usage <usage>`_, and how to it is `organized <organization>`_. Other pages
present a literature review of the field and the documentation of the source
code of the library.

.. toctree::
    :glob:
    :maxdepth: 1

    literature_review
    source/modules


.. _overview:

Overview
--------

This library contains prediction models as well as modules that automate tasks
required to training models, such as:

- data download,
- structure encoding,
- data formatting into training and test sets,
- result evaluation, and
- result visualization.

The `literature review <_lit_review>`_ page presents research in the field and
the list of references that were used to build the project.


.. _usage:

Usage
-----

**Install** the library with the following command:

::

   pip install diurnal


.. _organization:

Project Organization
--------------------

The project comprises the following subdirectories:

- ``demo`` : Python scripts that illustrate the usage of the library.
- ``docs`` : Documentation file written with ``Sphinx``.
- ``diurnal`` : Library source code.
- ``test`` : Automated test framework written with ``pytest``.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
