diurnal
=======

- `English (en) <#Predict-RNA-Secondary-Structures>`_
- `Français (fr) <#Prédire-des-structures-secondaires-dARN>`_

Predict RNA Secondary Structures
--------------------------------

This project aims at predicting RNA secondary structures with neural networks.
It comprises the following subdirectories.

- ``demo`` : Python scripts that illustrate the usage of the library.
- ``docs`` : Documentation file written with ``Sphinx``.
- ``diurnal`` : Library source code.
- ``test`` : Automated test framework written with ``pytest``.

Installation
````````````

Execute the following commands to install the library with pip.

.. code-block:: bash

   cd diurnal
   pip install .

You can also install the library with setup script.

Linux :

.. code-block:: bash

   pip install -r requirements.txt
   python3 setup.py install

Windows :

.. code-block:: bash

   pip install -r requirements.txt
   py setup.py install

Documentation
``````````````

The following commands generate the documentation. The Sphinx library has to be
installed.

.. code-block:: bash

   cd docs
   make html

Demonstration Scripts
`````````````````````

The directory  ``./demo`` comprises commented usages of the library.

Dataset Obtention
`````````````````

The library allows you to download the following datasets:

- ArchiveII
- RNASTRalign
- RNA_STRAND

Execute the following Python script to download the data:

.. code-block:: python

   import diurnal.database as db
   db.download_all("./data/")

Test Framwork
`````````````

The repository contains an automated test framework developed with the
``pytest`` library. Run the following commands to execute it.

.. code-block:: python

   cd test
   pytest


.. _Français - fr:

Prédire les structures secondaires de l'ARN
-------------------------------------------

Ce projet vise à prédire la structure secondaire de molécules d'ARN avec des
réseaux neuronaux. Il comprend les répertoires suivants :

- ``demo`` : Scripts Python qui illustrent l'utilisation de la bibliothèque.
- ``docs`` : Fichiers de documentation réalisés avec l'outil ``Sphinx``.
- ``diurnal`` : Fichiers sources des modèles. Le projet ùtilise le langage
  ``Python`` et la bibliothèque ``PyTorch``.
- ``test`` : Scripts utilisés pour valider le format des données et les
  modèles. Ils utilisent ``Python`` et la bibliothèque ``pytest``.

Installation
````````````

Exécutez les commandes suivantes pour installer la bibliothèque:

.. code-block:: bash

   cd diurnal
   pip install .

Il est aussi possible d'installer la bibliothèque avec le script ``setup.py``.

Linux :

.. code-block:: bash

   pip install -r requirements.txt # Installer les outils requis.
   python3 setup.py install # Installer la bibliothèque diurnal.

Windows :

.. code-block:: bash

   pip install -r requirements.txt # Installer les outils requis.
   py setup.py install # Installer la bibliothèque diurnal.

Documentation
``````````````

La commande suivante génère la documentation. La bibliothèque Sphinx doit être
installée.

.. code-block:: bash

   cd docs
   make html

Scripts de démonstration
````````````````````````

Consultez le répertoire ``./demo`` pour voir des exemples commentés
d'utilisation de la bibliothèque.

Obtenir l'ensemble de données
`````````````````````````````

La bibliothèque permet de télécharger et de décompresser trois ensembles de
données :

- ArchiveII
- RNASTRalign
- RNA_STRAND

Exécutez le script suivant pour obtenir les données :

.. code-block:: python

   import diurnal.database as db
   db.download_all("./data/")

Cadre de tests
``````````````

Le dépôt contient un cadre de tests automatisés développé avec la bibliothèque
``pytest``. Lancez les commandes suivantes pour l'exécuter.

.. code-block:: bash

   cd test
   pytest
