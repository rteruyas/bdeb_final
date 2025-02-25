# Pipeline de Machine Learning pour la Maintenance Prédictive (dataset AI4I)

![Maintenance](https://img.shields.io/badge/Maintenance-Active-green.svg)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-yellow?style=flat-square)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazonwebservices&logoColor=white)
![docker](https://img.shields.io/badge/docker-257bd6?style=for-the-badge&logo=docker&logoColor=white)


Un pipeline complet de machine learning de bout en bout pour la maintenance prédictive utilisant le jeu de données AI4I 2020.  
Collège Bois de Boulogne - 2025 

## Aperçu

Ce projet implémente un pipeline de machine learning prêt pour la production pour prédire les défaillances des machines en utilisant le jeu de données AI4I 2020 de maintenance prédictive. Le pipeline gère tout, de la préparation des données au déploiement du modèle, en mettant l'accent sur la reproductibilité, l'évolutivité et l'intégration continue.

### Caractéristiques Principales

- **Contrôle de Version des Données** : Versionnement complet des données et des modèles avec DVC
- **Pipeline CI/CD** : Tests automatisés et déploiement avec GitHub Actions
- **Infrastructure Cloud** : AWS S3 pour le stockage et la diffusion des modèles
- **Interface Interactive** : Tableau de bord Streamlit pour l'exploration des modèles
- **Service API** : Point d'accès FastAPI pour les prédictions en temps réel
- **Bonnes Pratiques MLOps** : Expériences reproductibles, suivi et surveillance des modèles

## Structure du Projet

```	
├── .github/                     			# Workflows GitHub Actions
│   └── workflows/                          
│       ├── register_and_upload_model.yml	# Enregistrer le meilleur modèle et le télécharger dans le stockage s3
├── config/                      			# Fichiers de configuration
│   ├── general.yaml             			# Configuration générale du projet
│   ├── preprocessing.yaml       			# Paramètres de prétraitement
│   ├── split_train_test.yaml    			# Paramètres de division train-test
│   └── train_model.yaml         			# Paramètres d'entraînement du modèle
├── data/                        			# Répertoire de données (suivi par DVC)
│   ├── raw_data.csv             			# Jeu de données original AI4I 2020
├── metrics/                     			# Métriques du modèle (suivies par DVC)
│   ├── model_metrics.json       			# Métriques de performance pour tous les modèles
│   └── training_status.json     			# Métadonnées d'entraînement
├── models/                      			# Modèles entraînés (suivis par DVC)
├── notebooks/                   			# Notebooks Jupyter pour l'exploration
├── src/                         			# Code source
│   ├── dashboard/               			# Tableau de bord Streamlit
│   │   └── app.py               			# Implémentation du tableau de bord
│   ├── utils/                   			# Fonctions utilitaires
│   │   └── path_utils.py        			# Gestion des chemins
│   ├── preprocessing.py         			# Pipeline de prétraitement des données
│   ├── split_train_test.py      			# Logique de division des données
│   └── train_model.py           			# Entraînement et évaluation des modèles
├── .dvcignore                   			# Fichiers à ignorer dans DVC
├── .gitignore                   			# Fichiers à ignorer dans Git
├── dvc.yaml                     			# Définition du pipeline DVC
├── dvc.lock                     			# Fichier de verrouillage du pipeline DVC
├── requirements.txt             			# Dépendances Python
└── README.md                    			# Ce fichier
```

## Jeu de Données

Le [Jeu de Données AI4I 2020 pour la Maintenance Prédictive](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) représente des données synthétiques qui imitent les données réelles de capteurs de maintenance prédictive d'équipements industriels. Il comprend :

- Mesures de processus (température, pression, etc.)
- Réglages des machines
- Types et indicateurs de défaillance
- Informations sur le type et l'âge des machines

L'objectif est de prédire quand les machines tomberont en panne, permettant une maintenance proactive pour éviter des temps d'arrêt coûteux.

## Premiers Pas

### Prérequis

- Python 3.9+ (recommandé 3.12)
- Git
- AWS CLI configuré avec les permissions appropriées
- Docker (optionnel, pour le déploiement conteneurisé)

### Installation

1. Cloner le dépôt :

```bash
git clone https://github.com/rteruyas/bdeb_final.git 
cd bdeb_final
```

2. Créer et activer un environnement virtuel :

```bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows : .venv\Scripts\activate
```

3. Installer les dépendances :

```bash
python -m pip install --upgrade pip 
pip install -r requirements.txt
```

4. Configurer DVC avec S3 :

```bash
dvc remote add -d s3remote s3://your-bucket-name/dvc-storage
```

5. Récupérer les données et les modèles :

```bash
dvc pull
```

### Exécuter le Pipeline

Pour exécuter le pipeline ML complet :

```bash
dvc repro
```

Cela va :
1. Prétraiter les données brutes
2. Diviser les données en ensembles d'entraînement et de test
3. Entraîner et évaluer plusieurs modèles
4. Suivre les métriques et les modèles dans mlflow (fourni par dagshub)

Pour exécuter une étape spécifique :

```bash
dvc repro <nom_de_l_étape>  # par exemple, dvc repro train
```

## Pipeline CI/CD

Ce projet utilise GitHub Actions pour CI/CD :

- **Intégration Continue** : Exécute des tests, du linting, et génère des métriques sur chaque pull request
- **Déploiement Continu** : Déploie automatiquement le dernier modèle et les services lorsque des changements sont fusionnés dans la branche principale

Les workflows sont définis dans `.github/workflows/`.

## Infrastructure AWS

Composants AWS clés :

- **S3** : Stocke les fichiers DVC, les modèles entraînés et les artefacts de déploiement
- **ECR** : Héberge les images Docker pour les services

## Développement

### Ajout de Nouveaux Modèles

1. Mettre à jour `config/train_model.yaml` avec la configuration de votre modèle
2. Ajouter l'implémentation du modèle à `src/train_model.py`
3. Exécuter `dvc repro train` pour entraîner et évaluer votre modèle

### Suivi des Expériences

Pour comparer différentes expériences :

```bash
dvc metrics show
dvc metrics diff
```

## Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## Remerciements

- Les créateurs du [Jeu de Données AI4I 2020 pour la Maintenance Prédictive](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
- [DVC](https://dvc.org/) pour le contrôle de version des données
- [FastAPI](https://fastapi.tiangolo.com/) pour le développement d'API
- [Streamlit](https://streamlit.io/) pour la création de tableaux de bord
