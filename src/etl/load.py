"""
ETL — Load : Chargement et préparation des données pour la modélisation.
"""
import os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import PROCESSED_DIR, TARGET_INDICATOR, RANDOM_STATE, TEST_SIZE


def load_processed_data() -> pd.DataFrame:
    """Charge les données transformées."""
    path = os.path.join(PROCESSED_DIR, 'energy_data_processed.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier non trouvé : {path}. Exécutez d'abord transform.py")
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame, target: str = TARGET_INDICATOR):
    """
    Prépare les features (X) et la cible (y) pour la modélisation.
    Exclut les colonnes non-numériques et la cible.
    """
    # Colonnes à exclure
    exclude = ['country_code', 'country_name', 'year', target]
    # Aussi exclure les features dérivées de la cible pour éviter le data leakage
    exclude += [c for c in df.columns if target in c and c != target]

    feature_cols = [c for c in df.columns if c not in exclude
                    and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    X = df[feature_cols].values
    y = df[target].values

    return X, y, feature_cols


def temporal_train_test_split(df: pd.DataFrame, target: str = TARGET_INDICATOR):
    """
    Split temporel : les années récentes pour le test.
    Plus réaliste qu'un split aléatoire pour les séries temporelles.
    """
    df_sorted = df.sort_values(['country_code', 'year'])

    # Seuil : dernières 20% des années comme test
    years = sorted(df_sorted['year'].unique())
    split_idx = int(len(years) * (1 - TEST_SIZE))
    train_years = years[:split_idx]
    test_years = years[split_idx:]

    train_df = df_sorted[df_sorted['year'].isin(train_years)]
    test_df = df_sorted[df_sorted['year'].isin(test_years)]

    X_train, y_train, feature_cols = prepare_features(train_df, target)
    X_test, y_test, _ = prepare_features(test_df, target)

    print(f"   Train : {len(train_df)} échantillons ({min(train_years)}-{max(train_years)})")
    print(f"   Test  : {len(test_df)} échantillons ({min(test_years)}-{max(test_years)})")
    print(f"   Features : {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, feature_cols, train_df, test_df


if __name__ == '__main__':
    print("Chargement et préparation des données...")
    df = load_processed_data()
    X_train, X_test, y_train, y_test, features, _, _ = temporal_train_test_split(df)
    print(f"\n✅ Données prêtes pour la modélisation")
    print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")
