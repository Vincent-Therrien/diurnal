.. diurnal documentation master file, created by
   sphinx-quickstart on Mon Apr 24 21:50:08 2023.

.. Generate the source code documentation file with `sphinx-apidoc -o ./source ../diurnal`

diurnal
=======

``diurnal`` is a Python library that helps to predict RNA secondary structures.

This library aims at streamlining the process of elaborating, training, and
validating predictive models. Researchers can use it under the permissive MIT
license to develop and publish models that can be easily replicated by other
users.

This page presents an :ref:`overview` of the project and its
:ref:`usage`. The :ref:`user-guide` provides detailed explanations of
the library. The :ref:`source-code-documentation` presents the signature
of all the components of the library. The :ref:`lit_review` provides the list of
references that were used to develop the project and discusses other similar
projects. The :ref:`developer-guide` explains how the project is organized and
how to contribute to it.

.. toctree::
    :glob:
    :maxdepth: 1

    user_guide
    source/modules
    developer_guide
    literature_review


.. _overview:

Overview
--------

This library contains RNA secondary structure predictive models and utility
components that automate training-related tasks.

In short, RNA secondary structure describes the *pairings* of nucleotides. RNA
(ribonucleic acid) is a molecule that performs a variety of biological
functions. It is made of a chain of *nucleotides* that can fold onto itself. One
can describe the structure of RNA molecules in different ways.

- The sequence of nucleotides is the *primary structure*.
- The way that nucleotides combine with one another is the
  *secondary structure*.
- The 3D arrangement of the molecule is the *tertiary structure*.

Determining the function of an RNA molecule from its primary structure is
difficult, so researchers rely on its secondary structure. Unfortunately,
determining secondary structures experimentally is costly and time-consuming.
There is therefore an interest in reliably determining secondary structures from
primary structures to understand the function of RNA molecules more effectively.

The ``diurnal`` library **predicts secondary structures from primary
structures**. It can:

- download RNA structure datasets,
- encode the datasets into trainable representations,
- prepare the data for different evaluation methods,
- train and evaluate models, and
- visualize results.

``diurnal`` is released under the MIT license and developed in Python. It relies
on the ``Numpy`` and ``PyTorch`` libraries for data manipulation and neural
network utilization.


.. _usage:

Basic Usage
-----------

**Install** the library with the following command:

.. code-block:: bash

   pip install diurnal

The code snippet below illustrates how to use the library to fetch data, train
a CNN-based model, and evaluate the model. More advanced usage is explained in
the :ref:`user-guide`. You can test this example by copying it in a file and
executing it as a Python script.

.. code-block:: python

   import torch
   from diurnal import database, train, visualize
   import diurnal.models
   from diurnal.models.networks import cnn

   SIZE = 512  # Maximum length of molecules used for training.

   # Download the `archiveII` dataset in the `data/` directory.
   database.download("./data/", "archiveII")
   # Format the raw data from the `archiveII` dataset into Numpy vectors.
   database.format(
       "./data/archiveII",  # Directory of the raw data to format.
       "./data/formatted",  # Formatted data output directory.
       SIZE  # Normalized size.
   )

   # Use the `5s` family as the test set and others as the training set.
   test_set, other_data = train.load_inter_family("./data/formatted", "5s")
   # Divide the training set into training and validation sets.
   train_set, validate_set = train.split_data(other_data, [0.8, 0.2])

   # Create a predictive model.
   model = diurnal.models.NN(
       model=cnn.Pairings_1,
       N=SIZE,
       n_epochs=3,
       optimizer=torch.optim.Adam,
       loss_fn=torch.nn.MSELoss,
       optimizer_args={"eps": 1e-4},
       loss_fn_args=None,
       verbosity=1)
   model.train(train_set)  # Train the model.

   f = model.test(test_set)  # Test the model.
   print(f"Average F1-score: {sum(f)/len(f):.4}")


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
