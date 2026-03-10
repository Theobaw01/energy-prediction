"""
ETL — Transform : Nettoyage, enrichissement et feature engineering.

Pipeline de transformation multi-niveaux :
1. Pivot des indicateurs (long → wide)
2. Interpolation intelligente des valeurs manquantes
3. Feature engineering avancé (lags, tendances, ratios dérivés)
4. Calcul de scores composites (lien énergie-développement)
"""
import os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import (
    RAW_DIR, PROCESSED_DIR, COUNTRIES, FOCUS_COUNTRY,
    TARGET_INDICATOR, INDICATOR_GROUPS
)


def load_raw_data() -> pd.DataFrame:
    """Charge les données brutes extraites."""
    path = os.path.join(RAW_DIR, 'energy_data_raw.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Données non trouvées : {path}\n"
            f"→ Exécutez d'abord : python src/etl/extract.py"
        )
    return pd.read_csv(path)


def pivot_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivote les données : une ligne par (pays, année), une colonne par indicateur.
    Élimine les doublons via aggfunc='first'.
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
    Gestion multi-stratégie des valeurs manquantes :
    - Interpolation linéaire par pays (pour série temporelle)
    - Forward/backward fill pour les extrémités
    - Médiane du groupe UEMOA pour les pays très incomplets
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'year']

    # 1. Interpolation linéaire par pays
    for country in df['country_code'].unique():
        mask = df['country_code'] == country
        df.loc[mask, numeric_cols] = (
            df.loc[mask, numeric_cols]
            .interpolate(method='linear', limit_direction='both')
        )

    # 2. Forward/backward fill résiduel
    for country in df['country_code'].unique():
        mask = df['country_code'] == country
        df.loc[mask, numeric_cols] = (
            df.loc[mask, numeric_cols].ffill().bfill()
        )

    # 3. Médiane régionale pour les colonnes entièrement vides
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering temporel :
    - Tendance normalisée
    - Variations annuelles (%)
    - Moyennes mobiles 3 et 5 ans
    - Lags (t-1, t-2)
    """
    features = df.copy()

    # Tendance temporelle normalisée [0, 1]
    year_min, year_max = features['year'].min(), features['year'].max()
    if year_max > year_min:
        features['year_norm'] = (features['year'] - year_min) / (year_max - year_min)
    else:
        features['year_norm'] = 0

    # Pour chaque indicateur numérique, créer des features dérivées
    base_indicators = [c for c in features.select_dtypes(include=[np.number]).columns
                       if c not in ['year', 'year_norm'] and '_' not in c[-4:]]

    # Limiter aux indicateurs principaux pour éviter l'explosion de features
    key_indicators = [
        'EG.USE.ELEC.KH.PC', 'EG.ELC.ACCS.ZS', 'NY.GDP.PCAP.CD',
        'FP.CPI.TOTL.ZG', 'SP.POP.TOTL', 'SH.DYN.MORT',
        'SP.DYN.LE00.IN', 'SP.URB.TOTL.IN.ZS'
    ]
    key_indicators = [k for k in key_indicators if k in features.columns]

    for col in key_indicators:
        grp = features.groupby('country_code')[col]

        # Variation annuelle (%)
        features[f'{col}_chg'] = grp.pct_change() * 100

        # Moyennes mobiles
        features[f'{col}_ma3'] = grp.transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        features[f'{col}_ma5'] = grp.transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

        # Lags
        features[f'{col}_lag1'] = grp.shift(1)
        features[f'{col}_lag2'] = grp.shift(2)

    return features


def add_derived_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Indicateurs dérivés à forte valeur analytique :
    - Gap électrique urbain/rural
    - Intensité énergétique (énergie/PIB)
    - Score de développement composite
    - Corrélation énergie-santé
    """
    derived = df.copy()

    # Gap d'accès électrique urbain/rural (indicateur clé développement)
    if 'EG.ELC.ACCS.UR.ZS' in derived.columns and 'EG.ELC.ACCS.RU.ZS' in derived.columns:
        derived['gap_elec_urbain_rural'] = (
            derived['EG.ELC.ACCS.UR.ZS'] - derived['EG.ELC.ACCS.RU.ZS']
        )

    # Intensité énergétique (kWh consommé par dollar de PIB/hab)
    if 'EG.USE.ELEC.KH.PC' in derived.columns and 'NY.GDP.PCAP.CD' in derived.columns:
        derived['intensite_energetique'] = (
            derived['EG.USE.ELEC.KH.PC'] /
            derived['NY.GDP.PCAP.CD'].replace(0, np.nan)
        )

    # Balance commerciale simplifiée (exports - imports)
    if 'NE.EXP.GNFS.ZS' in derived.columns and 'NE.IMP.GNFS.ZS' in derived.columns:
        derived['balance_commerciale'] = (
            derived['NE.EXP.GNFS.ZS'] - derived['NE.IMP.GNFS.ZS']
        )

    # Score énergie-santé (accès électricité vs mortalité infantile inversée)
    if 'EG.ELC.ACCS.ZS' in derived.columns and 'SH.DYN.MORT' in derived.columns:
        # Plus l'accès augmente et la mortalité baisse, meilleur est le score
        mort_max = derived['SH.DYN.MORT'].max()
        if mort_max > 0:
            derived['score_energie_sante'] = (
                derived['EG.ELC.ACCS.ZS'] *
                (1 - derived['SH.DYN.MORT'] / mort_max)
            )

    # Pression démographique sur l'énergie (population * conso)
    if 'SP.POP.TOTL' in derived.columns and 'EG.USE.ELEC.KH.PC' in derived.columns:
        derived['demande_elec_totale'] = (
            derived['SP.POP.TOTL'] * derived['EG.USE.ELEC.KH.PC'] / 1e6
        )  # en millions de kWh

    return derived


def add_country_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classement annuel des pays UEMOA par indicateur clé.
    Utile pour le dashboard BCEAO (benchmark).
    """
    ranked = df.copy()

    rank_indicators = ['EG.USE.ELEC.KH.PC', 'EG.ELC.ACCS.ZS',
                       'NY.GDP.PCAP.CD', 'SH.DYN.MORT']
    rank_indicators = [r for r in rank_indicators if r in ranked.columns]

    for col in rank_indicators:
        ascending = True if col == 'SH.DYN.MORT' else False  # Mortalité: bas = mieux
        ranked[f'{col}_rank'] = ranked.groupby('year')[col].rank(
            ascending=not ascending, method='min'
        )

    return ranked


def transform():
    """Pipeline de transformation complet."""
    print("=" * 70)
    print("  TRANSFORMATION DES DONNÉES — Togo & Zone UEMOA")
    print("=" * 70)

    # 1. Charger
    print("\n  1. Chargement des données brutes...")
    raw = load_raw_data()
    print(f"     {len(raw):,} enregistrements, "
          f"{raw['indicator_code'].nunique()} indicateurs, "
          f"{raw['country_code'].nunique()} pays")

    # 2. Pivoter
    print("\n  2. Pivot (1 ligne = 1 pays × 1 année)...")
    pivoted = pivot_indicators(raw)
    print(f"     {len(pivoted)} lignes × {len(pivoted.columns)} colonnes")

    # 3. Valeurs manquantes
    print("\n  3. Gestion des valeurs manquantes...")
    missing_before = pivoted.isnull().sum().sum()
    cleaned = handle_missing_values(pivoted)
    missing_after = cleaned.isnull().sum().sum()
    print(f"     Valeurs manquantes : {missing_before} → {missing_after}")

    # 4. Features temporelles
    print("\n  4. Feature engineering temporel...")
    featured = add_temporal_features(cleaned)
    print(f"     {len(featured.columns)} colonnes après features temporelles")

    # 5. Indicateurs dérivés
    print("\n  5. Indicateurs dérivés (gap, scores, ratios)...")
    enriched = add_derived_indicators(featured)
    print(f"     {len(enriched.columns)} colonnes après enrichissement")

    # 6. Rankings
    print("\n  6. Classement UEMOA...")
    final = add_country_ranking(enriched)

    # 7. Nettoyage final
    final = final.replace([np.inf, -np.inf], np.nan)
    final = final.fillna(0)

    # 8. Sauvegarder
    output_path = os.path.join(PROCESSED_DIR, 'energy_data_processed.csv')
    final.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\n  {'─' * 50}")
    print(f"  Dimensions finales : {final.shape[0]} lignes × {final.shape[1]} colonnes")
    print(f"  Fichier : {output_path}")

    # Stats Togo
    togo = final[final['country_code'] == FOCUS_COUNTRY]
    if not togo.empty:
        print(f"\n  ── Focus Togo ──")
        for col, label in [
            ('EG.USE.ELEC.KH.PC', 'Conso. élec.'),
            ('EG.ELC.ACCS.ZS', 'Accès élec.'),
            ('gap_elec_urbain_rural', 'Gap urb./rur.'),
            ('NY.GDP.PCAP.CD', 'PIB/hab'),
            ('SH.DYN.MORT', 'Mort. infantile'),
        ]:
            if col in togo.columns:
                latest = togo[togo['year'] == togo['year'].max()][col].values
                if len(latest) > 0:
                    print(f"     {label:20s} : {latest[0]:.1f}")

    print(f"\n  ✓ Transformation terminée.")
    return final


if __name__ == '__main__':
    transform()
