diurnal
=======

- `English (en) <#Predict-RNA-Secondary-Structures>`_
- `Français (fr) <#Prédire-les-structures-secondaires-de-lARN>`_


Predict RNA Secondary Structures
--------------------------------

This project is a Python library that can predict RNA secondary structures with
neural networks.

The project requires Pytorch, which can be installed as described on the page
https://pytorch.org/get-started/locally/.


Installation
````````````

Using an active Python virtual environment in which Pytorch is installed,
**install** the diurnal library with the following command:

.. code-block:: bash

   cd diurnal
   pip install .


Documentation
````````````

The **documentation** can be obtained by installing the requirement packages
with the command:

.. code-block:: bash

   pip install -r docs/requirements.txt

and generating the content with the following commands:

.. code-block:: bash

   cd docs
   make html


Demonstrations
````````````

The directory ``demo`` contains usage examples of the library. You can execute
them as Python script, as shown below:

.. code-block:: bash

   python3 demo/baseline.py # On Linux
   py demo/baseline.py # On Windows


Test Suite
````````````

The **test suite** can be executed with the command:

.. code-block:: bash

   pytest test


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

En utilisant un environnement Python actif dans lequel la bibliothèque PyTorch
est installée, exécutez les commandes suivantes pour installer la bibliothèque:

.. code-block:: bash

   cd diurnal
   pip install .

Il est aussi possible d'installer la bibliothèque avec le script ``setup.py``.

Linux :

.. code-block:: bash

   pip install -r requirements.txt  # Installer les outils requis.
   python3 setup.py install  # Installer la bibliothèque diurnal.

Windows :

.. code-block:: bash

   pip install -r requirements.txt  # Installer les outils requis.
   py setup.py install  # Installer la bibliothèque diurnal.


Scripts de démonstration
````````````````````````

Consultez le répertoire ``./demo`` pour voir des exemples commentés
d'utilisation de la bibliothèque.


Documentation
``````````````

Installez les modules requis avec la commande

.. code-block:: bash

   pip install -r docs/requirements.txt

puis générez la documentation avec les commandes :

.. code-block:: bash

   cd docs
   make html


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

   pytest test
