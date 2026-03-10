# ⚡ Énergie & Développement — Togo (Zone UEMOA / BCEAO)

> Système complet d'analyse prédictive multi-cibles pour le Togo et la zone UEMOA, exploitant l'IA pour évaluer l'impact de l'énergie sur le développement humain.

## Description

Ce projet combine **Intelligence Artificielle**, **Open Data** et **Business Intelligence** pour analyser et prédire l'évolution de la consommation énergétique au Togo et dans les 8 pays de la zone UEMOA. Il met en évidence les liens entre accès à l'électricité, santé publique et croissance économique — des enjeux au cœur des missions de la **BCEAO**.

### Objectifs

- **Prédire** la consommation électrique par habitant au Togo (horizon 2024-2028)
- **Quantifier** l'impact de l'accès à l'énergie sur la mortalité infantile
- **Benchmarker** le Togo face aux autres pays UEMOA sur des indicateurs clés
- **Projeter** l'évolution du taux d'électrification avec intervalles de confiance

### Résultats Clés

| Cible | Meilleur Modèle | R² | MAE |
|---|---|---|---|
| Consommation électrique (kWh/hab) | Stacking Ensemble | 0.77 | ~20.75 |
| Mortalité infantile (‰) | LightGBM | 0.92 | ~0.82 |
| Accès électricité (%) | Ensemble | ~0.85 | ~1.45 |

**Impact estimé** : chaque point supplémentaire d'accès à l'électricité réduit la mortalité infantile de **1.45 points pour mille**.

## Sources de Données (Open Data)

| Source | Données | Indicateurs |
|---|---|---|
| **Banque Mondiale (WDI)** | 25 indicateurs × 8 pays UEMOA | Énergie, Économie, Santé, Démographie |

**Pays couverts** : 🇹🇬 Togo (focus), 🇸🇳 Sénégal, 🇧🇯 Bénin, 🇨🇮 Côte d'Ivoire, 🇧🇫 Burkina Faso, 🇲🇱 Mali, 🇳🇪 Niger, 🇬🇼 Guinée-Bissau

**Couverture temporelle** : 2000 – 2023 (4 651 observations extraites)

## Architecture du Projet

```
energy-prediction-gabon/
├── data/
│   ├── raw/                    # Données brutes API Banque Mondiale
│   ├── processed/              # Données transformées (192 × 78 features)
│   └── predictions/            # Prédictions historiques & projections
├── src/
│   ├── etl/
│   │   ├── extract.py          # Extraction API WDI (25 indicateurs)
│   │   ├── transform.py        # Pipeline 6 étapes + feature engineering
│   │   └── load.py             # Préparation multi-cibles
│   ├── models/
│   │   ├── train.py            # Entraînement (RF, GB, XGB, LGBM, Stacking)
│   │   └── predict.py          # Prédictions + projections 2024-2028
│   └── utils/
│       └── config.py           # Configuration centralisée
├── dashboard/
│   └── app.py                  # Dashboard Streamlit (5 onglets)
├── models/                     # Modèles sauvegardés (.joblib)
├── requirements.txt
└── README.md
```

## Technologies

| Catégorie | Technologies |
|---|---|
| **ETL** | Python, Pandas, NumPy, API REST (Banque Mondiale WDI) |
| **Machine Learning** | Scikit-learn (RandomForest, GradientBoosting, Stacking), XGBoost, LightGBM |
| **Feature Engineering** | Moyennes mobiles, lag features, indicateurs dérivés, rankings |
| **Visualisation** | Plotly (graphiques interactifs) |
| **Dashboard BI** | Streamlit (5 onglets, KPIs, projections) |
| **Persistance** | joblib (modèles), CSV (données) |

## Pipeline

### 1. Extract — Collecte de données

Extraction automatisée de **25 indicateurs** via l'API Banque Mondiale pour **8 pays UEMOA** :

| Groupe | Indicateurs |
|---|---|
| **Énergie** | Consommation électrique, accès total/urbain/rural, énergies renouvelables, émissions CO2 |
| **Économie** | PIB/hab, croissance, inflation, dette, IDE, balance commerciale |
| **Santé** | Mortalité infantile, espérance de vie, dépenses santé, accès eau potable |
| **Démographie** | Population totale, croissance, urbanisation |

### 2. Transform — Feature Engineering

Pipeline en **6 étapes** :
1. Pivot des indicateurs en colonnes
2. Traitement des valeurs manquantes (interpolation + forward/backward fill + médiane)
3. Features temporelles (variation %, MA3, MA5, lag1, lag2)
4. Indicateurs dérivés (gap électrification urbain/rural, intensité énergétique, score énergie-santé)
5. Rankings par pays
6. Normalisation et validation

→ **192 lignes × 78 colonnes** après transformation

### 3. Train — Modélisation Multi-Cibles

3 cibles prédites indépendamment :

| Cible | Description |
|---|---|
| `energy` | Consommation électrique par habitant (kWh) |
| `health` | Mortalité infantile (pour 1 000 naissances) |
| `access` | Taux d'accès à l'électricité (%) |

**Modèles** : RandomForest, GradientBoosting, XGBoost, LightGBM + **Stacking Ensemble** (meta-learner Ridge)

**Validation** : Split temporel (pas de fuite de données) + TimeSeriesSplit cross-validation

### 4. Predict — Projections Futures

- Prédictions historiques sur le jeu de test
- **Projections 2024-2028** avec intervalles de confiance (±2σ croissant)
- Analyse d'impact énergie ↔ développement

### 5. Dashboard BI — 5 onglets interactifs

| Onglet | Contenu |
|---|---|
| ⚡ **Énergie** | Évolution consommation, accès, gap urbain/rural, émissions CO2 |
| 🏥 **Impact Social** | Corrélation énergie ↔ mortalité, accès eau, espérance de vie |
| 💰 **Économie** | PIB, croissance, dette, IDE vs énergie |
| 🤖 **Prédictions** | Historique + projections 2024-2028 avec intervalles de confiance |
| 🌍 **Benchmark UEMOA** | Radar comparatif, classements, dernières valeurs |

## Installation

```bash
# Cloner le projet
git clone https://github.com/Theobaw01/energy-prediction-gabon.git
cd energy-prediction-gabon

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

```bash
# 1. Extraction des données (API Banque Mondiale)
python src/etl/extract.py

# 2. Transformation & feature engineering
python src/etl/transform.py

# 3. Entraînement des modèles (3 cibles × 5 algorithmes)
python src/models/train.py

# 4. Génération des prédictions & projections 2024-2028
python src/models/predict.py

# 5. Lancement du dashboard interactif
streamlit run dashboard/app.py
```

Le dashboard sera accessible sur **http://localhost:8501**.

## Contexte BCEAO / UEMOA

Ce projet s'inscrit dans les missions de la **Banque Centrale des États de l'Afrique de l'Ouest** en matière de :
- **Analyse de données macroéconomiques** de la zone UEMOA
- **Modélisation prédictive** pour appuyer les décisions de politique économique
- **Suivi des indicateurs de développement** (énergie, santé, économie)
- **Intelligence artificielle appliquée** à l'analyse des données régionales

## Auteur

**Theodore Bawana** — Développeur IA  
[Portfolio](https://github.com/theobaw01) | [GitHub](https://github.com/theobaw01)
