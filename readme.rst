diurnal
=======

- `English - en`_
- `Français - fr`_

.. _English - en:

Predict RNA Secondary Structure with DL and Thermodynamics
----------------------------------------------------------

This project aims at predicting RNA secondary structures by combining neural
networks and thermodynamic models. Reinforcement learning is planned to be
included to make that combination more effective.


.. _Français - fr:

Prédire des structures secondaires d'ARN avec l'AP et la thermodynamique
------------------------------------------------------------------------

Ce projet vise à prédire la structure secondaire de molécules d'ARN en
combinant des réseaux neuronaux avec des modèles thermodynamiques. On prévoit
utiliser l'apprentissage par renforcement pour améliorer la performance de
l'outil. Le projet comprend les répertoires suivants :

- ``data`` : Ensemble de donnée qui regroupe des descriptions de molécules
  d'ARN.
- ``docs`` : Fichiers de documentation réalisés avec l'outil ``Sphinx``.
- ``models`` : Modèles déjà entraînés qui peuvent être utilisés pour effectuer
  des prédictions.
- ``src`` : Fichiers sources des modèles. Le projet ùtilise le langage
  ``Python`` et la bibliothèque ``PyTorch``.
- ``test`` : Scripts utilisés pour valider le format des données et les
  modèles. Ils utilisent ``Python`` et la bibliothèque ``unittest``.

Configuration
`````````````

Le langage ``Python`` doit être installé sur le système (version minimale :
3.7. version recommandée : 3.11). Une carte graphique NVIDIA est recommandée
pour entraîner les réseaux neuronaux

Créez d'abord un environnement virtual ``Python`` et installez les outils
nécessaires.

Linux :

.. code-block:: bash

   python3 -m venv ./venv # Créer l'environnement virtuel
   source ./venv/bin/activate # Activer l'environnement
   pip install -r requirements.txt # Installer les outils requis

Windows :

.. code-block:: bash

   python -m venv .\venv # Créer l'environnement virtuel
   .\venv\Scripts\activate.bat # Activer l'environnement
   pip install -r requirements.txt # Installer les outils requis

Visualisation des données
`````````````````````````

Vous pouvez utiliser l'outil ``draw_rna`` (https://github.com/DasLab/draw_rna)
pour lancer l'exemple ``test/visualize_rna``.

Obtenir l'ensemble de données
`````````````````````````````

Décompressez l'archive ``data/archiveII.tar.gz`` inclue dans le dépôt.
L'archive a été obtenue du projet ``dl-rna`` à l'adresse
https://github.com/marcellszi/dl-rna/releases.

Exécuter les tests
``````````````````

Pour exécuter un test, lancer les commandes suivantes :

Linux :

.. code-block:: bash

   source ./venv/bin/activate
   python3 ./test/<NOM_DU_TEST>

Windows :

.. code-block:: bash

   .\venv\Scripts\activate.bat
   python .\test\<NOM_DU_TEST>

où ``<NOM_DU_TEST>`` est l'une des options suivantes :

- ``pytorch_validation/cnn_example.py`` : Validation de l'installation de
  ``pytorch`` avec un réseau convolutionnel simple.
- ``dataset_validation.py`` : Vérification du dépaquetage de l'ensemble de
  données.
