# Prevision de la Demande Electrique -- Zone UEMOA (Horizon 2045)

**Projet Data Engineering et Developpement en Intelligence Artificielle**

Pipeline ETL et IA complet -- 8 pays -- 82 features -- 1990-2023 vers 2045 -- Maitrise de l'analyse de donnees

---

## Resume

Conception d'un **pipeline end-to-end** (Extract-Transform-Load et modelisation) pour anticiper la demande electrique des **8 pays de l'UEMOA** a l'horizon 2045 :

- **Extraction automatisee** de 21 indicateurs macroeconomiques via l'API REST Banque Mondiale (8 pays, 1990-2023)
- **Ingenierie de 82 features** : lags temporels, moyennes mobiles, transformations log, ratios demographiques, interactions croisees, encodage pays (one-hot)
- **Entrainement et comparaison de 7 algorithmes** avec cross-validation temporelle (5 folds) ; meilleur modele retenu : **Ridge Regression** (alpha=10.0, log-transform de la cible, R²=0.968)
- **Dashboard interactif single-page** (Streamlit, Plotly) : selection de n'importe quel pays UEMOA, benchmark comparatif 8 pays, projections avec intervalles de confiance a 95%

---

## Resultats

| Metrique | Valeur |
|---|---|
| Meilleur modele | Ridge Regression (alpha=10.0) |
| R² (test) | 0.968 |
| Variable cible | log1p(conso_totale_gwh) -- evaluation en GWh via expm1 |
| Donnees d'entrainement | 272 observations (8 pays x 34 ans) |
| Features | 82 variables construites a partir de 21 indicateurs bruts |
| Validation | Split temporel 80/20 + cross-validation temporelle 5 folds |
| Projections | 2024-2045 pour les 8 pays UEMOA |

### Comparaison des 7 modeles

| Modele | Description |
|---|---|
| **Ridge Regression** (retenu) | alpha=10.0, log1p target, one-hot country encoding |
| Random Forest | 200 arbres, profondeur 10 |
| Gradient Boosting | 200 arbres, profondeur 5, lr 0.05 |
| XGBoost | 200 arbres, profondeur 6 |
| LightGBM | 200 arbres, profondeur 6 |
| Linear Regression | Baseline lineaire |
| Stacking Ensemble | RF + GB + XGB + LGBM, meta-modele Ridge |

Les metriques exactes (R², RMSE, MAE, MAPE) sont recalculees a chaque execution du pipeline et affichees dans le dashboard.

---

## Sources de Donnees

| Source | Volume | Couverture |
|---|---|---|
| Banque Mondiale (WDI) | environ 5 600 observations brutes | 21 indicateurs x 8 pays x 34 ans |

**Pays UEMOA couverts** : Togo, Senegal, Benin, Cote d'Ivoire, Burkina Faso, Mali, Niger, Guinee-Bissau

**Periode** : 1990 -- 2023

### 21 Indicateurs par domaine

| Domaine | Indicateurs |
|---|---|
| Demographie (7) | Population totale, croissance, urbanisation, fecondite, esperance de vie, pop 0-14 ans, pop 15-64 ans |
| Energie (6) | kWh/hab, acces total/urbain/rural, renouvelable (%), energie kg petrole/hab |
| Economie (5) | PIB/hab, PIB total, croissance PIB, industrie (% PIB), inflation |
| Social (3) | Abonnements mobile, alphabetisation, chomage |

---

## Architecture du Projet

```
energy-prediction-gabon/
├── data/
│   ├── raw/                          # Observations brutes (API Banque Mondiale)
│   ├── processed/                    # Features transformees (82 colonnes)
│   └── predictions/                  # Predictions + projections 2024-2045
├── src/
│   ├── etl/
│   │   ├── extract.py                # Extraction API WDI (21 indicateurs, 8 pays)
│   │   ├── transform.py              # Feature engineering (82 variables)
│   │   └── load.py                   # Split temporel, log-transform, one-hot encoding
│   ├── models/
│   │   ├── train.py                  # 7 modeles + CV temporelle + feature importance
│   │   └── predict.py                # Projections 2024-2045, 8 pays, IC 95%
│   └── utils/
│       └── config.py                 # Configuration centralisee (indicateurs, pays)
├── dashboard/
│   └── app.py                        # Dashboard Streamlit single-page
├── models/
│   ├── model_energy.joblib           # Modele Ridge sauvegarde
│   ├── results.csv                   # Metriques comparatives des 7 modeles
│   ├── cv_scores.csv                 # Scores de cross-validation par fold
│   └── feature_importance.csv        # Variables les plus influentes
├── .streamlit/
│   └── config.toml                   # Configuration Streamlit (theme, port)
├── requirements.txt
└── README.md
```

---

## Technologies

| Categorie | Technologies |
|---|---|
| ETL et Data | Python, Pandas, NumPy, API REST Banque Mondiale (WDI) |
| Machine Learning | Scikit-learn (Ridge, RandomForest, GradientBoosting, Stacking), XGBoost, LightGBM |
| Validation | TimeSeriesSplit (5 folds), split temporel 80/20, feature importance |
| Feature Engineering | Lags (t-1, t-2), MA (3, 5), log-transforms, ratios, one-hot country encoding |
| Visualisation | Plotly (graphiques interactifs, theme plotly_white) |
| Dashboard | Streamlit (single-page, selecteur multi-pays, benchmark UEMOA) |
| Versioning | Git, GitHub |

---

## Pipeline ETL et Modelisation

### 1. Extract -- Collecte automatisee

Extraction de **21 indicateurs** via l'API REST Banque Mondiale pour **8 pays UEMOA** (1990-2023).

Resultat : environ 5 600 observations brutes extraites automatiquement.

### 2. Transform -- Feature Engineering

Pipeline de transformation en plusieurs etapes :

1. Pivot des indicateurs en colonnes (format tabulaire)
2. Imputation des valeurs manquantes (interpolation + forward/backward fill + mediane)
3. Variable cible : conso_totale_gwh = Population x kWh/hab / 1 000 000
4. Features temporelles : variation annuelle (%), lags (t-1, t-2), moyennes mobiles (MA3, MA5)
5. Features classiques : population active, ratio de dependance, PIB/hab calcule, interactions
6. Features avancees : log(population), log(PIB), PIB/actif, GWh/PIB, population electrifiee, industrie absolue, interaction croissance PIB x population
7. Encodage one-hot des pays (country encoding)
8. Validation et nettoyage final

Resultat : 272 lignes x 82 colonnes apres transformation.

### 3. Load -- Preparation des donnees

- Application de log1p sur la variable cible (stabilisation de la variance)
- Split temporel 80/20 (pas de fuite de donnees)
- Encodage one-hot des pays pour le modele

### 4. Train -- Modelisation

**Cible** : log1p(conso_totale_gwh), evaluation en GWh via expm1

**Strategie** : Entrainement sur les 8 pays UEMOA simultanement (272 observations).

**7 algorithmes compares** :
- Linear Regression (baseline)
- Ridge Regression (alpha=10.0) -- retenu (R²=0.968)
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Stacking Ensemble (RF + GB + XGB + LGBM, meta Ridge)

**Validation** :
- Split temporel (80/20)
- Cross-validation temporelle (TimeSeriesSplit, 5 folds)
- Export automatique des scores CV et de l'importance des features

### 5. Predict -- Projections 2045

- Predictions historiques sur l'ensemble du jeu de donnees (8 pays)
- Projections 2024-2045 (22 ans) pour chaque pays UEMOA avec intervalles de confiance a 95%
- Methode hybride : 60% ML + 40% CAGR
- Resume avec MAE/MAPE par pays et tableau de projection finale

### 6. Dashboard -- Visualisations et Interpretations

Dashboard single-page avec selecteur de pays dans la barre laterale.

| Section | Contenu |
|---|---|
| En-tete et KPI | Metriques cles du pays selectionne (population, demande, acces, projection, precision) |
| Pipeline | 4 cartes resumant les etapes Extract, Transform, Train, Predict |
| Demande et modeles | Evolution GWh + comparaison des 7 algorithmes |
| Population et acces | Co-evolution population/demande, acces electrique |
| Variables et fiabilite | Feature importance + cross-validation temporelle |
| Validation | Valeurs observees vs predictions du modele |
| Projection 2045 | Trajectoire historique + projections avec IC 95% |
| Comparaison 8 pays | Benchmark regional UEMOA |
| Sources et donnees | 21 liens API exacts, donnees brutes, traitees, predictions, projections, resultats modeles |

Chaque section est accompagnee d'une **interpretation analytique**.

---

## Installation

```bash
# Cloner le projet
git clone https://github.com/Theobaw01/energy-prediction-togo.git
cd energy-prediction-togo

# Creer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / Mac
# .\venv\Scripts\Activate.ps1   # Windows

# Installer les dependances
pip install -r requirements.txt
```

## Utilisation

```bash
# 1. Extraction des donnees (API Banque Mondiale)
python src/etl/extract.py

# 2. Transformation et feature engineering
python src/etl/transform.py

# 3. Entrainement des modeles (7 algorithmes + cross-validation)
python src/models/train.py

# 4. Generation des predictions et projections 2024-2045 (8 pays)
python src/models/predict.py

# 5. Lancement du dashboard interactif
python -m streamlit run dashboard/app.py
```

Le dashboard sera accessible sur **http://localhost:8501**.

---

## Competences demontrees

| Competence | Application dans le projet |
|---|---|
| Pipeline ETL / ELT | Extraction API, transformation, chargement automatise |
| Machine Learning | 7 modeles compares, Ridge retenu (R²=0.968) |
| Analyse predictive | Projections 2024-2045 avec intervalles de confiance |
| Python et librairies IA | Scikit-learn, XGBoost, LightGBM, Pandas, NumPy |
| Tableaux de bord (BI) | Dashboard Streamlit + Plotly, graphiques interactifs |
| Modelisation de donnees | 82 features construites a partir de 21 indicateurs bruts |
| Validation rigoureuse | TimeSeriesSplit 5 folds, split temporel, feature importance |

---

## Auteur

**Theodore Bawana** -- Developpeur en Intelligence Artificielle

GitHub : https://github.com/Theobaw01
