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
| 3       | Neighboring nucleotide in the 5' direction (i.e. upstream)        |
+---------+-------------------------------------------------------------------+
| 4       | Neighboring nucleotide in the 3' direction (i.e. downstream)      |
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
   primary_onehot = Primary.to_vector(primary)

   # Encode the list of pairings into a one-hot vector. For instance, if
   # `secondary` contains the value `[2, -1, 0]`, the encoded structure will be
   # `[[1, 0, 0], [0, 1, 0], [0, 0, 1]]`, which correspond to `(.)`.
   secondary_onehot = Secondary.to_vector(secondary)

   # Obtain the list of bases from an encoded vector.
   primary = Primary.to_bases(primary_onehot)

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
       structure.Primary.to_vector,  # Primary to vector map.
       structure.Secondary.to_vector  # Secondary to vector map.
   )

Executing this function will generate the following files:

- ``families.npy``: Encoded RNA families.
- ``readme.rst``: Metadata such as the file creation date.
- ``names.txt``: The list of molecule names.
- ``primary_structures.npy``: Encoded primary structures.
- ``secondary_structures.npy``: Encoded secondary structures.

The ``.npy`` files can be read with the function ``numpy.load(filename)``.


Prepare Data for Training
^^^^^^^^^^^^^^^^^^^^^^^^^

 Formatted data can be loaded and split for training. In the context of RNA
 secondary structure prediction, there are two main ways to divide data:

- In **inter-family testing**, molecules of one RNA molecule are used as the
  test set and all the other molecules of the dataset are used for training.
  The point of this type of training is to measure how well the model can
  predict the structure of unfamiliar molecules.
- In **intra-family testing**, molecules of the dataset are randomly sampled to
  elaborate the test set. Since the model is trained with all families of
  molecules, it can better predict secondary structures than in inter-family
  testing.

The code snippet below shows how to load data for the two types of testing:

.. code-block:: python

   from diurnal import train

   # Inter-family testing.
   # Load formatted data for inter-family testing with the `5s` family.
   test_set, train_set = train.load_inter_family("./data/formatted", "5s")

   # Intra-family testing.
   # Load all formatted data and randomize the order of molecules.
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

the model makes random predictions. When using bracket notation (``(``, ``.``,
``)``), the model with generate random sequences of the three possible
characters. The F1-score of predictions should be at around 0.3. This can be
useful to compare performances with other models and ensure that the data
processing pipeline works well.


Evaluate Results
^^^^^^^^^^^^^^^^


Save and Load Models
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Write the model into the `saved_model` directory.
   model.save("saved_model")

   # Erase the model to clear space.
   del model

   # Load the model stored in the `saved_model` directory.
   loaded_model = diurnal.models.NN(
       cnn.Pairings_1,
       SIZE,
       3,
       torch.optim.Adam,
       torch.nn.MSELoss,
       {"eps": 1e-4},
       None,
       verbosity=1)
   loaded_model.load("saved_model")

   # Test the model. Should obtain the same result as before.
   f = loaded_model.test(test_set)
   print(f"Average F1-score of the saved model: {sum(f)/len(f):.4}")

   # Visualize an example of a prediction.
   print(f"\nSample prediction from the test set (`{test_set['names'][0]}`).")
   p = test_set["primary_structures"][0]
   s = test_set["secondary_structures"][0]
   visualize.prediction(p, s, loaded_model.predict(p))


References
----------

.. bibliography:: references.bib
