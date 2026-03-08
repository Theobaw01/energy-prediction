# Modèle Prédictif de Consommation Énergétique — Gabon

> **Note** : Ce projet a été développé en local dans le cadre de mes travaux personnels en Data Science et rendu public récemment pour démonstration.

## Description

Système complet d'analyse et de prédiction de la consommation énergétique au Gabon, avec pipeline ETL automatisé, modèles de Machine Learning et tableaux de bord interactifs.

**Contexte** : Le Gabon, avec Libreville comme principal centre urbain (~50% de la population, ~70% de la consommation électrique), fait face à des défis énergétiques croissants. Ce projet exploite les données publiques de la Banque Mondiale pour analyser les tendances et prédire la consommation future.

## Sources de Données (Open Data)

| Source | Données | Accès |
|---|---|---|
| **Banque Mondiale** | Consommation électrique, accès énergie, PIB, population | [API publique](https://api.worldbank.org/v2/) |
| **IEA** | Mix énergétique, production, émissions CO2 | [iea.org/countries/gabon](https://www.iea.org/countries/gabon) |
| **EnergyData.info** | Centrales électriques, réseau de transmission | [energydata.info](https://energydata.info/dataset?q=gabon) |

## Architecture du Projet

```
energy-prediction/
├── data/
│   ├── raw/               # Données brutes (Banque Mondiale, IEA)
│   ├── processed/         # Données nettoyées et transformées
│   └── predictions/       # Résultats des prédictions
├── src/
│   ├── etl/               # Pipeline ETL/ELT
│   │   ├── extract.py     # Extraction des données
│   │   ├── transform.py   # Nettoyage et transformation
│   │   └── load.py        # Chargement des données
│   ├── models/            # Modèles ML
│   │   ├── train.py       # Entraînement des modèles
│   │   ├── evaluate.py    # Évaluation et métriques
│   │   └── predict.py     # Prédictions
│   └── utils/             # Utilitaires
│       └── config.py      # Configuration
├── notebooks/
│   └── exploration.ipynb  # Analyse exploratoire
├── dashboard/
│   └── app.py             # Dashboard Streamlit
├── models/                # Modèles sauvegardés
├── requirements.txt
└── README.md
```

## Technologies

| Catégorie | Technologies |
|---|---|
| **ETL/ELT** | Python, Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **Visualisation** | Matplotlib, Seaborn, Plotly |
| **Dashboard BI** | Streamlit |
| **Données** | Banque Mondiale Open Data, IEA |

## Pipeline ETL

1. **Extract** : Collecte automatisée des données de consommation énergétique via l'API Banque Mondiale
2. **Transform** : Nettoyage, gestion des valeurs manquantes, feature engineering (tendances, saisonnalité, indicateurs économiques)
3. **Load** : Stockage structuré des données prêtes pour l'analyse et la modélisation

## Modèles Prédictifs

- **Random Forest** : Modèle de référence
- **XGBoost** : Optimisation par gradient boosting
- **LightGBM** : Performance et rapidité

Métriques : RMSE, MAE, R² — avec validation croisée temporelle.

## Dashboard BI

Dashboard interactif Streamlit avec :
- Visualisation des tendances de consommation par pays
- Comparaisons régionales (zone UEMOA)
- Prédictions futures avec intervalles de confiance
- Indicateurs clés (KPIs)

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# 1. Exécuter le pipeline ETL
python src/etl/extract.py
python src/etl/transform.py

# 2. Entraîner les modèles
python src/models/train.py

# 3. Lancer le dashboard
streamlit run dashboard/app.py
```

## Auteur

**Theodore Bawana** — Développeur IA  
[Portfolio](https://github.com/theobaw01) | [GitHub](https://github.com/theobaw01)
