diurnal
=======

- `English (en) <#Predict-RNA-Secondary-Structure>`_
- `Français (fr) <#Prédire-des-structures-secondaires-d'ARN>`_

Predict RNA Secondary Structure
-------------------------------

This project aims at predicting RNA secondary structures by combining neural
networks and thermodynamic models. Reinforcement learning is planned to be
included to make that combination more effective.

- ``demo`` : Python scripts that illustrate the usage of the library.
- ``docs`` : Documentation file written with ``Sphinx``.
- ``models`` : Trained models.
- ``diurnal`` : Library source code.
- ``test`` : Automated test framework written with ``pytest``.

Install with pip
````````````````

Execute the following instruction to install the library with pip.

.. code-block:: bash

   cd diurnal
   pip install .

Demonstration Scripts
`````````````````````

The directory  ``./demo`` comprises commented usages of the library.

Alternative Installation
````````````````````````

You can also install the library with ``venv``.

Linux :

.. code-block:: bash

   python3 -m venv ./venv
   source ./venv/bin/activate
   pip install -r requirements.txt
   py setup.py install

Windows :

.. code-block:: bash

   py -m venv .\venv
   .\venv\Scripts\activate.bat
   pip install -r requirements.txt
   py setup.py install

Dataset Obtention
`````````````````````````````

The library allows you to downlaod the following datasets:

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
``pytest`` library. Launch the following command to run it.

Linux :

.. code-block:: python

   python3 pytest

Windows :

.. code-block:: python

   py pytest


.. _Français - fr:

Prédire des structures secondaires d'ARN
----------------------------------------

Ce projet vise à prédire la structure secondaire de molécules d'ARN avec des
réseaux neuronaux. Le projet comprend les répertoires suivants :

- ``demo`` : Scripts Python qui illustrent l'utilisation de la bibliothèque.
- ``docs`` : Fichiers de documentation réalisés avec l'outil ``Sphinx``.
- ``models`` : Modèles déjà entraînés qui peuvent être utilisés pour effectuer
  des prédictions.
- ``diurnal`` : Fichiers sources des modèles. Le projet ùtilise le langage
  ``Python`` et la bibliothèque ``PyTorch``.
- ``test`` : Scripts utilisés pour valider le format des données et les
  modèles. Ils utilisent ``Python`` et la bibliothèque ``pytest``.

Installation avec pip
`````````````````````

Exécutez la commande suivante pour installer la bibliothèque:

.. code-block:: bash

   cd diurnal
   pip install .

Scripts de démonstration
````````````````````````

Consultez le répertoire ``./demo`` pour voir des exemples commentés
d'utilisation de la bibliothèque.

Installation alternative
````````````````````````

Il est aussi possible d'installer la bibliothèque avec ``venv``.

Linux :

.. code-block:: bash

   python3 -m venv ./venv # Créer l'environnement virtuel.
   source ./venv/bin/activate # Activer l'environnement.
   pip install -r requirements.txt # Installer les outils requis.
   py setup.py install # Installer la bibliothèque diurnal.

Windows :

.. code-block:: bash

   py -m venv .\venv # Créer l'environnement virtuel.
   .\venv\Scripts\activate.bat # Activer l'environnement.
   pip install -r requirements.txt # Installer les outils requis.
   py setup.py install # Installer la bibliothèque diurnal.

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
``pytest``. Lancez la commande suivante pour l'exécuter.

Linux :

.. code-block:: python

   python3 pytest

Windows :

.. code-block:: python

   py pytest

