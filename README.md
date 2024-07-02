Projet M2DEV - Groupe 1
=======================

Description
-----------

Ce projet est développé par le Groupe 1 dans le cadre du cours de IA. Il utilise FastAPI pour créer une API web rapide et efficace. Les principales dépendances incluent FastAPI, Uvicorn, pandas, joblib, requests, et scikit-learn.

Installation
------------

### Prérequis

-   Python 3.12 ou version supérieure
-   pip (gestionnaire de paquets Python)

### Étapes d'installation

1.  Clonez le dépôt du projet :

    bash

    Copier le code

    `git clone <URL_DU_DEPOT>
    cd <NOM_DU_DEPOT>`

2.  Installez les dépendances :

    bash

    Copier le code

    `pip install -r requirements.txt`

Utilisation
-----------

### Lancer le serveur

Pour démarrer le serveur FastAPI, utilisez la commande suivante :

bash

Copier le code

`uvicorn main:app --reload`

Où `main` est le nom du fichier Python contenant l'application FastAPI (`app` est l'instance de FastAPI).

### Points de terminaison API

Vous pouvez accéder à la documentation interactive de l'API une fois le serveur lancé via :

`http://127.0.0.1:8000/docs`



Dépendances
-----------

-   FastAPI
-   [Uvicorn](https://www.uvicorn.org/)
-   pandas
-   [joblib](https://joblib.readthedocs.io/)
-   requests
-   scikit-learn


### Étapes pour contribuer :

1.  Fork le dépôt
2.  Créez votre branche de fonctionnalité (`git checkout -b feature/AmazingFeature`)
3.  Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4.  Poussez vers la branche (`git push origin feature/AmazingFeature`)
5.  Ouvrez une Pull Request
