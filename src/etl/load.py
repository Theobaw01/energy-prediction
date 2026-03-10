"""
ETL — Load : Préparation des données pour la modélisation multi-cibles.

Prépare les features et cibles pour 3 axes de prédiction :
  1. Consommation électrique (kWh/hab)
  2. Mortalité infantile (lien énergie-santé)
  3. Taux d'accès à l'électricité (projection)
"""
import os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import (
    PROCESSED_DIR, TARGET_INDICATOR, RANDOM_STATE,
    TEST_SIZE, PREDICTION_TARGETS, FOCUS_COUNTRY
)


def load_processed_data() -> pd.DataFrame:
    """Charge les données transformées."""
    path = os.path.join(PROCESSED_DIR, 'energy_data_processed.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Données non trouvées : {path}\n"
            f"→ Exécutez d'abord : python src/etl/transform.py"
        )
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame, target: str = TARGET_INDICATOR):
    """
    Prépare X (features) et y (cible) en excluant :
    - Colonnes non-numériques
    - La cible elle-même
    - Les features dérivées de la cible (éviter data leakage)
    """
    exclude = ['country_code', 'country_name', 'year', target]
    # Exclure features directement dérivées de la cible
    exclude += [c for c in df.columns if c.startswith(target) and c != target]

    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    X = df[feature_cols].values
    y = df[target].values

    return X, y, feature_cols


def temporal_train_test_split(df: pd.DataFrame, target: str = TARGET_INDICATOR):
    """
    Split temporel : années récentes en test.
    Plus réaliste qu'un split aléatoire pour les séries temporelles.
    """
    # Filtrer les lignes avec cible valide
    df_valid = df[df[target].notna() & (df[target] != 0)].copy()
    df_valid = df_valid.sort_values(['country_code', 'year'])

    years = sorted(df_valid['year'].unique())
    split_idx = int(len(years) * (1 - TEST_SIZE))
    train_years = years[:split_idx]
    test_years = years[split_idx:]

    train_df = df_valid[df_valid['year'].isin(train_years)]
    test_df = df_valid[df_valid['year'].isin(test_years)]

    X_train, y_train, feature_cols = prepare_features(train_df, target)
    X_test, y_test, _ = prepare_features(test_df, target)

    print(f"     Train : {len(train_df)} obs. ({min(train_years)}-{max(train_years)})")
    print(f"     Test  : {len(test_df)} obs. ({min(test_years)}-{max(test_years)})")
    print(f"     Features : {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, feature_cols, train_df, test_df


def prepare_all_targets(df: pd.DataFrame) -> dict:
    """
    Prépare les données pour tous les axes de prédiction définis
    dans PREDICTION_TARGETS.

    Returns:
        dict avec clé = nom cible, valeur = (X_train, X_test, y_train, y_test, features, ...)
    """
    results = {}

    for target_key, target_info in PREDICTION_TARGETS.items():
        indicator = target_info['indicator']
        if indicator not in df.columns:
            print(f"  ⚠ Indicateur {indicator} absent — saut de {target_key}")
            continue

        print(f"\n  ▸ Cible : {target_info['name']}")
        try:
            data = temporal_train_test_split(df, indicator)
            results[target_key] = {
                'X_train': data[0],
                'X_test': data[1],
                'y_train': data[2],
                'y_test': data[3],
                'feature_names': data[4],
                'train_df': data[5],
                'test_df': data[6],
                'info': target_info,
            }
        except Exception as e:
            print(f"    ✗ Erreur pour {target_key}: {e}")

    return results


def get_focus_country_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait uniquement les données du pays focus (Togo)."""
    return df[df['country_code'] == FOCUS_COUNTRY].copy()


if __name__ == '__main__':
    print("  Chargement et préparation des données...\n")
    df = load_processed_data()

    print("  ── Split par cible de prédiction ──")
    targets = prepare_all_targets(df)

    print(f"\n  ✓ {len(targets)} cibles préparées pour la modélisation")
    for key, data in targets.items():
        print(f"    • {key}: X_train {data['X_train'].shape}, "
              f"X_test {data['X_test'].shape}")
