.. diurnal documentation master file, created by
   sphinx-quickstart on Mon Apr 24 21:50:08 2023.

diurnal
=======

This project aims at simplifying RNA secondary structure prediction by providing
a set of routines that automate commun tasks among prediction tools.

.. toctree::
   :maxdepth: 1
   :caption: Library Documentation:

   modules

Main Features
-------------

The ``diurnal`` library can:

- Download and unwrap three RNA structure datasets (ArchiveII, RNASTRalign, and
  RNA STRAND)
- Convert raw data file into matrix representation.
- Simplify the training of pyTorch models.
- Evaluate predictions.
- Save models.
- Visualize result performances.

Usage Example
-------------

The example below illustrates the use of the library:

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader
   
   import diurnal.database as db
   from diurnal.transform import PrimaryStructure as s1
   from diurnal.transform import SecondaryStructure as s2
   from diurnal.transform import Family as f
   from diurnal import train, evaluate
   from diurnal.models import DiurnalBasicModel
   from diurnal.networks import cnn as diurnalCNN
   
   # Download the dataset.
   db.download("./data/", "archiveII")
   
   # Format the dataset into numpy `.npy` files.
   db.format(
       "./data/archiveII", # Directory of the raw data to format.
       "./data/formatted", # Formatted data output directory.
       512, # Normalized size.
       s1.iupac_to_onehot, # RNA encoding scheme.
       s2.pairings_to_onehot, # RNA encoding scheme.
       f.onehot
   )

   # Load data from `.npy` files.
   data, names = train.load_data("./data/formatted/")
   
   # Split the data in training and test sets.
   train_set, test_set = train.split_data(data, [0.8, 0.2])
   
   # Create the RNA secondary structure prediction model.
   model = DiurnalBasicModel(
           diurnalCNN.RNA_CNN_classes, [512],
           torch.optim.Adam, [1e-04],
           torch.nn.MSELoss()
       )
   
   model.train(DataLoader(train_set, batch_size=32), 5)
   f1 = model.test(DataLoader(test_set, batch_size=32), evaluate.three_class_f1)
   
   # Display performances.
   evaluate.summarize_results(f1, "CNN, Three-class evaluation")

The directory ``diurnal/demo`` contains commented scripts that explain how to
use the library.

References
----------

**Models**:

- CNNFold: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04540-7
- ATTFold: https://www.frontiersin.org/articles/10.3389/fgene.2020.612086/full#B17
- UFold: https://academic.oup.com/nar/article/50/3/e14/6430845
- MXFolx2: https://www.nature.com/articles/s41467-021-21194-4
- CDPFold: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6540740/

**Datasets**:

- RNAStralign: https://www.urmc.rochester.edu/rna/
- RNAalign: https://drive.google.com/drive/folders/19KPRYJjjMJh1qdMhtmUoYA_ncw3ocAHc
- RFam: https://rfam.org/

**Comments**:

- Szikszai: https://academic.oup.com/bioinformatics/article/38/16/3892/6617348

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
