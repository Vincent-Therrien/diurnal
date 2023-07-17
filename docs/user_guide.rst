.. _user-guide:

User Guide
==========

This page presents how to use the ``diurnal`` library to predict RNA secondary
structures.

List of Modules
---------------

The ``diurnal`` library comprises the following modules. The ``numpy`` and
``scikit-learn`` libraries must be installed in the Python virtual environment
to use them.

- `diurnal.database <source/diurnal.html#module-diurnal.database>`__: Download
  and use RNA datasets.
- `diurnal.evaluate <source/diurnal.html#module-diurnal.evaluate>`__: Evaluate
  the performances of predictive models.
- `diurnal.family <source/diurnal.html#module-diurnal.family>`__: Manipulate
  data related to RNA families.
- `diurnal.structure <source/diurnal.html#module-diurnal.structure>`__: Format
  RNA structures into different representations.
- `diurnal.train <source/diurnal.html#module-diurnal.train>`__: Prepare data
  for training models.
- `diurnal.visualize <source/diurnal.html#module-diurnal.visualize>`__:
  Visualize data with ``matplotlib`` graphs or output in the terminal.

The library also contains the `diurnal.models <source/diurnal.models.html>`__
subpackage that comprises predictive models. The
`pytorch <https://pytorch.org/>`_ library must be installed to use this
package.

Usage
-----

Download Data
^^^^^^^^^^^^^

The code snippet below illustrates how to download RNA datasets:

.. code-block:: python

   from diurnal import database

   # `data` is the directory in which data is downloaded. `archiveII` is the
   # name of the dataset.
   database.download("./data/", "archiveII")

This code will download the `archiveII` dataset and write the raw data into the
`data` directory. If the directory does not exist, it is created. Three
datasets can be downloaded:

- `archiveII`: 3975 molecules. Originally retrieved from
  `<https://rna.urmc.rochester.edu/pub/archiveII.tar.gz>`_ :cite:`rochester`.
- `RNASTRalign`: 37149 molecules. Originally retrieved from
  `<https://www.urmc.rochester.edu/rna/>`_ :cite:`rochester`.
- `RNA_STRAND`: 4666 molecules. Originally retrieved from
  `<http://www.rnasoft.ca/strand/>`_ :cite:`rnastrand`.

One may instead use the function ``diurnal.database.download_all()`` to
download all available datasets.

Molecules are represented in these datasets with CT files, which is a
tab-separated format. The first line of a CT file contains (1) the number of
nucleotides in the molecule and (2) the name of the molecule. The other lines
of the files all follow the same syntax, which is presented in the table below:

+---------+-------------------------------------------------------------------+
| Column  | Content                                                           |
+=========+===================================================================+
| 1       | Index of the nucleotide (starts at 1)                             |
+---------+-------------------------------------------------------------------+
| 2       | Nucleotide (A, C, G, or U)                                        |
+---------+-------------------------------------------------------------------+
| 3       | Neighboring nucleotide index in the 5' direction (i.e. upstream)  |
+---------+-------------------------------------------------------------------+
| 4       | Neighboring nucleotide index in the 3' direction (i.e. downstream)|
+---------+-------------------------------------------------------------------+
| 5       | Index of the paired nucleotide. If the nucleotide is unpaired,    |
|         | the value ``0`` is used.                                          |
+---------+-------------------------------------------------------------------+
| 6       | Index of the nucleotide (same as column 1)                        |
+---------+-------------------------------------------------------------------+

One can read the content of CT files with the following code:

.. code-block:: python

   # Read the first molecule of the `archiveII` dataset.
   filename = "./data/archiveII/5s_Acanthamoeba-castellanii-1.ct"
   name, primary, secondary = diurnal.utils.rna_data.read_ct_file(filename)

which returns the following data:

- The **name** of the molecule.
- The **primary structure** of the molecule, represented as a list of
  ``A``, ``C``, ``G``, and ``U`` characters.
- The **secondary structure** of the molecule, represented as a list of
  pairings in which the *i* th element of the list is paired with the index
  that it contains (for instance, ``(((...)))`` can be represented as
  ``[8, 7, 6, -1, -1, -1, 2, 1, 0]``). Since Python uses zero-based indices,
  ``-1`` is used for unpaired nucleotides instead of ``0`` like CT files do.


Format Data
^^^^^^^^^^^

The RNA structure data of CT files must be converted into numerical vectors to
train predictive models. The module ``diurnal.structure`` can encode the data
into other formats, as shown below:

.. code-block:: python

   from diurnal.structure import Primary, Secondary

   # Encode the list of bases into a one-hot vector. For instance, if `primary`
   # contains the value `['A', 'C']`, the encoded structure will be
   # `[[1, 0, 0, 0], [0, 1, 0, 0]]`.
   primary_onehot = Primary.to_onehot(primary)

   # Encode the list of pairings into a one-hot vector. For instance, if
   # `secondary` contains the value `[2, -1, 0]`, the encoded structure will be
   # `[[1, 0, 0], [0, 1, 0], [0, 0, 1]]`, which correspond to `(.)`.
   secondary_onehot = Secondary.to_onehot(secondary)

   # Obtain the list of bases from an encoded vector.
   primary = Primary.to_sequence(primary_onehot)

   # Obtain the bracket notation from an encoded vector.
   bracket = Secondary.to_bracket(secondary_onehot)

For convenience, the library can encode a whole dataset of CT files into
another representation and store them in Numpy files. Users can subsequently
read these already-formatted files instead of reading CT files every time. The
following code snippet shows how to do that:

.. code-block:: python

   from diurnal import database, structure

   database.format(
       "./data/archiveII",  # Directory of the raw data to format.
       "./data/formatted",  # Formatted data output directory.
       512,  # Normalized size. Short molecules are padded at the 3' end.
       structure.Primary.to_onehot,  # Primary to vector map.
       structure.Secondary.to_onehot  # Secondary to vector map.
   )

Executing this function will generate the following files:

- ``families.npy``: Encoded RNA families.
- ``readme.rst``: Metadata such as the file creation date.
- ``names.txt``: The list of molecule names.
- ``primary_structures.npy``: Encoded primary structures.
- ``secondary_structures.npy``: Encoded secondary structures.

The ``.npy`` files can be read with the function ``numpy.load(filename)``,
which returns a ``numpy.array`` object.


Prepare Data for Training
^^^^^^^^^^^^^^^^^^^^^^^^^

Formatted data can be loaded and split for training. In the context of RNA
secondary structure prediction, there are a few ways to divide data:

- In **inter-family testing** (also called *family-wise cross-validation* by
  Sato et al. :cite:`mxfold2`), the model is trained and tested with datasets
  that comprise different RNA families. Therefore, training and testing data
  are structurally different. The point of this type of training is to measure
  how well the model can predict the structure of unfamiliar molecules.
- In **sequence testing** (also called *sequence-wise cross-validation*
  by Sato et al. :cite:`mxfold2`), the model is trained and tested with datasets
  that comprise the same RNA families. Therefore, training and testing data
  are structurally similar. Consequently, this type of testing is expected to
  yield more accurate results than inter-family testing.
- In **intra-family testing**, models are trained and tested with RNA molecules
  that belong to the same family. Therefore, training and testing data
  are structurally very similar and results are expected to yield more accurate
  results than sequence testing. This type of testing does not appear to be
  discussed in published work, but it can be useful to validate models.

The code snippet below shows how to load data for inter-family and sequence
testing.

.. code-block:: python

   from diurnal import train, family

   # Inter-family testing.
   test_set = train.load_families("./data/formatted", "5s")
   train_set = train.load_families("./data/formatted", family.all_but("5s"))

   # Sequence testing.
   data = train.load_data("./data/formatted", randomize = True)
   # Divide data in training (80 % of points) and test sets (20 % of points).
   train_set, test_set = train.split_data(data, [0.8, 0.2])

One may also divide data to perform K-fold validation, as shown below:

.. code-block:: python

   from diurnal import train

   # Do five K-fold splits.
   K = 5
   data = train.load_data("./data/formatted", randomize = True)
   for i in range(K)
       train_set, test_set = train.k_fold_split(data, [0.8, 0.2], K, i)
       # Train and test a model for this K-split.


Train Models
^^^^^^^^^^^^

One can load predictive models comprised within the ``diurnal`` library, as
demonstrated in the code snippet below:

.. code-block:: python

   import torch
   from diurnal import models

   # Load a `diurnal` neural network based on the `Pairings_1` architecture.
   model = models.NN(
       model=models.networks.cnn.Pairings_1,
       N=SIZE,
       n_epochs=3,
       optimizer=torch.optim.Adam,
       loss_fn=torch.nn.MSELoss,
       optimizer_args={"eps": 1e-4},
       loss_fn_args=None,
       verbosity=1)
   # Train the model
   model.train(train_set)

In the example above, the class ``diurnal.models.NN`` is a wrapper around a
``pytorch`` neural network. Another type of ``diurnal`` models are baselines,
which make basic predictions. For instance, in the code below,

.. code-block:: python

   from diurnal.models import baseline

   model = baseline.Random()
   model.train(train_set)

the model makes random predictions. This can be useful to compare performances
with other models and ensure that the data processing pipeline works well.


Predict Structures
^^^^^^^^^^^^^^^^^^

You can predict structures as shown below:

.. code-block:: python

   from diurnal import structure

   # Assume that `model` is a trained `diurnal.model` object. The method
   # `predict` accepts primary structures encoded in the same format
   # that was used for training (in this case, one-hot encoding).
   primary_structure = list("AAAACCCCUUUU")
   encoded_primary_structure = structure.Primary.to_onehot(primary_structure)
   prediction = model.predict(encoded_primary_structure)

The data format returned by the ``predict`` method depends on the architecture
of the ``model`` object. For example, a model may return a one-hot encoded
bracket notation of the secondary structure.


Evaluate Results
^^^^^^^^^^^^^^^^

The are two main ways to evaluate secondary structure predictions.

The first and most widespread method consists in using the **recall** and
**precision** :cite:`cnnfold` :cite:`mxfold2` :cite:`attfold` :cite:`ufold`
:cite:`cdpfold`. This evaluation method uses the following metrics:

- True positives (TP): number of paired bases that are correctly predicted to
  be paired.
- True negative (TN): number of unpaired bases that are correctly predicted to
  be unpaired.
- False positives (FP): number of paired bases that are erroneously predicted
  to be unpaired.
- False negatives (FN): number of unpaired bases that are erroneously predicted
  to be paired.

Recall (or *true positive rate* or *sensitivity*) is the probability that a
positive prediction is actually positive. It is computed with the following
equation:

.. math::

    recall = \frac{TP}{TP + FN}

Precision (or *positive predictive value*) is the fraction of relevant elements
among retrieved elements. It is computed with the following equation:

.. math::

    precision = \frac{TP}{TP + FP}

The geometric mean of these two values is the **F1-score**, which is
also called *F1*, *F1-measure*, *F-score*, or *F-measure*:

.. math::

    F1 = 2 \times \frac{recall \times precision}{recall + precision}

These evaluation metrics can be computed with the function
``diurnal.evaluate.recall_precision_f1``, as shown below:

.. code-block:: python

   from diurnal import evaluate

   true = list("(((....)))")
   prediction = list("((......))")
   r, p, f1 = evaluate.recall_precision_f1(true, prediction)

One drawback of these evaluation metrics is that they do not make a distinction
about whether a nucleotide is paired with a base in the 5' or 3' direction.
Therefore, when comparing the structures ``(((....)))`` and ``)))....(((``, the
precision, recall, and f1-score all have a perfect value of 1 even though the
predicted structure is inaccurate. The ``diurnal`` library therefore uses
another metric, the *micro f1-score*, which generalizes precision and recall to
classification problems that rely on more than two classes. One can also obtain
the confusion matrix of predicted structures. The code snippet below
shows how to compute the micro f1-score and confusion matrix:

.. code-block:: python

   from diurnal import evaluate

   true = list("(((....)))")
   prediction = list("((......))")
   micro_f1 = evaluate.micro_f1(true, prediction)
   confusion_matrix = evaluate.get_confusion_matrix(true, prediction)


Save Models
^^^^^^^^^^^

Predictive models can be written in files for subsequent reuse, as shown below:

.. code-block:: python

   # Assume that `model` is a trained `diurnal.model` object.
   model.save(directory = "saved_model")

In addition to writing the model in the provided directory, the library also
generates:

- a file containing the list of the names of the molecules that were used for
  training the model, and
- an informative file containing metadata.


Load Models
^^^^^^^^^^^

Predictive models can be loaded from saved files, as shown below:

.. code-block:: python

   from diurnal import models

   model = models.NN(
      cnn.Pairings_1,
      SIZE,
      None,
      torch.optim.Adam,
      torch.nn.MSELoss,
      {"eps": 1e-4},
      None,
      verbosity=1)
   model.load("saved_model")


Visualize Results
^^^^^^^^^^^^^^^^^

The module ``diurnal.visualize`` contains utility functions that can help users
visualize results with graphs or console output.


Elaborate Predictive Models
---------------------------

The class ``diurnal.models.Basic`` represents a basic predictive model. One may
derive this class to create a new predictive model. Four methods need to be
implemented in the derived class:

- ``_train(data)``: Train the model.
- ``_predict(primary)``: Predict and return a secondary structure.
- ``_save(directory)``: Write the model in files.
- ``_load(directory)``: Read the model from files.

The class ``diurnal.models.NN`` is an example of a predictive model that
is derived from the class ``diurnal.models.Basic``. The ``NN`` class is used
to represent predictive models based on neural networks.


References
----------

.. bibliography:: references.bib
