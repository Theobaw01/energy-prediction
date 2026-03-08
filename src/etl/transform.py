"""
ETL — Transform : Nettoyage, uniformisation et feature engineering des données énergétiques.
"""
import os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import RAW_DIR, PROCESSED_DIR, COUNTRIES, TARGET_INDICATOR


def load_raw_data() -> pd.DataFrame:
    """Charge les données brutes extraites."""
    path = os.path.join(RAW_DIR, 'energy_data_raw.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier non trouvé : {path}. Exécutez d'abord extract.py")
    return pd.read_csv(path)


def pivot_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivote les données : une ligne par (pays, année), une colonne par indicateur.
    """
    pivoted = df.pivot_table(
        index=['country_code', 'country_name', 'year'],
        columns='indicator_code',
        values='value',
        aggfunc='first'
    ).reset_index()

    pivoted.columns.name = None
    return pivoted


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gère les valeurs manquantes :
    - Interpolation linéaire par pays
    - Forward/backward fill pour les extrémités
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'year']

    # Interpolation par pays
    for country in df['country_code'].unique():
        mask = df['country_code'] == country
        df.loc[mask, numeric_cols] = (
            df.loc[mask, numeric_cols]
            .interpolate(method='linear', limit_direction='both')
        )

    # Fill restant
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering :
    - Tendance temporelle
    - Variations annuelles
    - Moyennes mobiles
    - Indicateurs dérivés
    """
    features = df.copy()

    # Tendance temporelle
    features['year_normalized'] = (features['year'] - features['year'].min()) / \
                                   (features['year'].max() - features['year'].min())

    # Variations annuelles par pays et indicateur
    numeric_cols = [c for c in features.select_dtypes(include=[np.number]).columns
                    if c not in ['year', 'year_normalized']]

    for col in numeric_cols:
        # Variation annuelle (%)
        features[f'{col}_pct_change'] = (
            features.groupby('country_code')[col].pct_change() * 100
        )
        # Moyenne mobile 3 ans
        features[f'{col}_ma3'] = (
            features.groupby('country_code')[col]
            .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )

    # Remplacer inf et NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)

    return features


def transform():
    """
    Pipeline de transformation complet.
    """
    print("=" * 60)
    print("TRANSFORMATION DES DONNÉES")
    print("=" * 60)

    # 1. Charger les données brutes
    print("\n1. Chargement des données brutes...")
    raw = load_raw_data()
    print(f"   {len(raw)} enregistrements chargés")

    # 2. Pivoter les indicateurs
    print("\n2. Pivot des indicateurs (1 ligne = 1 pays × 1 année)...")
    pivoted = pivot_indicators(raw)
    print(f"   {len(pivoted)} lignes, {len(pivoted.columns)} colonnes")

    # 3. Gestion des valeurs manquantes
    print("\n3. Gestion des valeurs manquantes...")
    missing_before = pivoted.isnull().sum().sum()
    cleaned = handle_missing_values(pivoted)
    missing_after = cleaned.isnull().sum().sum()
    print(f"   Valeurs manquantes : {missing_before} → {missing_after}")

    # 4. Feature engineering
    print("\n4. Feature engineering...")
    featured = add_features(cleaned)
    print(f"   {len(featured.columns)} features générées")

    # 5. Sauvegarder
    output_path = os.path.join(PROCESSED_DIR, 'energy_data_processed.csv')
    featured.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ Transformation terminée")
    print(f"   Fichier : {output_path}")
    print(f"   Dimensions : {featured.shape}")

    # Stats résumées
    print("\n--- Résumé par pays ---")
    for code, name in COUNTRIES.items():
        country_data = featured[featured['country_code'] == code]
        if not country_data.empty and TARGET_INDICATOR in country_data.columns:
            target_vals = country_data[TARGET_INDICATOR]
            print(f"   {name}: {len(country_data)} années, "
                  f"conso moy: {target_vals.mean():.1f} kWh/hab")

    return featured


if __name__ == '__main__':
    transform()
