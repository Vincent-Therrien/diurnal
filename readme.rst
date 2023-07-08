diurnal
=======

- `English (en) <#Predict-RNA-Secondary-Structures>`_
- `Français (fr) <#Prédire-les-structures-secondaires-de-lARN>`_


Predict RNA Secondary Structures
--------------------------------

This project is a Python library that can predict RNA secondary structures with
neural networks.

**Install** the library with the following command:

.. code-block:: bash

   pip install diurnal

The **documentation** can be built with the command:

.. code-block:: bash

   cd docs
   make html

The directory ``demo`` contains usage examples of the library. You can execute
them as Python script, as shown below:

.. code-block:: bash

   python3 demo/baseline.py # On Linux
   py demo/baseline.py # On Windows

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


Scripts de démonstration
````````````````````````

Consultez le répertoire ``./demo`` pour voir des exemples commentés
d'utilisation de la bibliothèque.


Documentation
``````````````

La commande suivante génère la documentation. La bibliothèque Sphinx doit être
installée.

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

   cd test
   pytest


Objectifs de développement
--------------------------

La liste suivante énumère des objectifs de développement du projet :

- [ ] Élaborer un carnet interactif (en : *notebook*) pour illustrer le
  fonctionnement de l'outil.
- [ ] Conteneuriser l'environnement de développement et des scripts de
  validation avec Docker pour reproduire les résultats automatiquement.
- [ ] Améliorer l'empaquetage des résultats (ex. : inclure les noms des
   molécules utilisées pour l'entraînement)
- [ ] Développer davantage les fonctionnalités de prédiction
  - [ ] Mieux décrire le fonctionnement des CNN
  - [ ] Réaliser des modèles basés sur les RNN
  - [ ] Réaliser des modèles basés sur les transformateurs
  - [ ] Réaliser des modèles basés sur les encodeurs / décodeurs
  - [ ] Utiliser des mécanismes récursifs pour appliquer des contraintes rigides
    sur les résultats.
  - [ ] Introduire des informations liées à la thermodynamique des molécules
    pour améliorer les prédictions.
  - [ ] Ajouter des couches multi-branches
- [ ] Investiguer l'utilisation de l'apprentissage par renforcement
- [ ] Déployer un service Web pour permettre à des utilisateurs de tester les
  modèle à partir d'un navigateur.
