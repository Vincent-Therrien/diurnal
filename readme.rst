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

Obtenir l'ensemble de données
`````````````````````````````


Exécuter les tests
``````````````````

Pour valider l'installation de ``PyTorch``, exécutez :

Linux :

.. code-block:: bash

   source ./venv/bin/activate
   cd test/pytorch_validation
   python3 ./cnn_example.py

Windows :

.. code-block:: bash

   .\venv\Scripts\activate.bat
   cd test\pytorch_validation
   python .\cnn_example.py
