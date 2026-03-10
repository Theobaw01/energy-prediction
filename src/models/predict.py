"""
Prédictions multi-cibles + Projections futures (2024-2028).

Génère :
1. Prédictions sur les données historiques (validation)
2. Projections futures avec intervalles de confiance
3. Analyse d'impact : lien énergie ↔ santé ↔ économie
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import (
    MODELS_DIR, PREDICTIONS_DIR, COUNTRIES, FOCUS_COUNTRY,
    PREDICTION_TARGETS, FORECAST_HORIZON
)
from etl.load import load_processed_data, prepare_features


def load_model(target_key: str) -> dict:
    """Charge un modèle sauvegardé pour une cible donnée."""
    path = os.path.join(MODELS_DIR, f'model_{target_key}.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modèle non trouvé : {path}\n"
            f"→ Exécutez d'abord : python src/models/train.py"
        )
    return joblib.load(path)


def predict_historical(df: pd.DataFrame, target_key: str) -> pd.DataFrame:
    """
    Génère les prédictions sur les données historiques.
    Utile pour évaluer la qualité des modèles visuellement.
    """
    saved = load_model(target_key)
    model = saved['model']
    scaler = saved['scaler']
    target_info = saved['target_info']
    indicator = target_info['indicator']

    X, y, _ = prepare_features(df, indicator)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)

    results = df[['country_code', 'country_name', 'year']].copy()
    results['target'] = target_key
    results['target_name'] = target_info['name']
    results['unit'] = target_info['unit']
    results['actual'] = y
    results['predicted'] = predictions
    results['error'] = results['actual'] - results['predicted']
    results['error_pct'] = np.where(
        results['actual'] != 0,
        (results['error'] / results['actual'] * 100).round(2),
        0
    )

    return results


def project_future(df: pd.DataFrame, target_key: str,
                   horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """
    Projections futures par pays avec intervalles de confiance.

    Méthode : on utilise les tendances récentes pour extrapoler les features,
    puis on applique le modèle entraîné.
    """
    saved = load_model(target_key)
    model = saved['model']
    scaler = saved['scaler']
    feature_names = saved['feature_names']
    target_info = saved['target_info']
    indicator = target_info['indicator']

    projections = []
    max_year = int(df['year'].max())

    for code, name in COUNTRIES.items():
        country_df = df[df['country_code'] == code].sort_values('year')
        if country_df.empty or len(country_df) < 5:
            continue

        # Utiliser les 5 dernières années pour estimer les tendances
        recent = country_df.tail(5)

        for h in range(1, horizon + 1):
            future_year = max_year + h
            future_row = {}

            # Extrapoler chaque feature par tendance linéaire
            for feat in feature_names:
                if feat in recent.columns:
                    vals = recent[feat].values
                    # Tendance linéaire simple
                    if len(vals) >= 2 and not np.all(np.isnan(vals)):
                        trend = np.polyfit(range(len(vals)), vals, 1)
                        future_row[feat] = trend[0] * (len(vals) - 1 + h) + trend[1]
                    else:
                        future_row[feat] = vals[-1] if len(vals) > 0 else 0
                else:
                    future_row[feat] = 0

            # Construire le vecteur de features
            X_future = np.array([[future_row.get(f, 0) for f in feature_names]])
            X_future = np.nan_to_num(X_future, nan=0.0, posinf=0.0, neginf=0.0)
            X_future_scaled = scaler.transform(X_future)

            pred = model.predict(X_future_scaled)[0]

            # Intervalle de confiance naïf basé sur la volatilité récente
            if indicator in recent.columns:
                recent_vals = recent[indicator].values
                std = np.std(recent_vals) if len(recent_vals) > 1 else 0
                ci_factor = 1.96 * std * np.sqrt(h)  # Croît avec l'horizon
            else:
                ci_factor = 0

            projections.append({
                'country_code': code,
                'country_name': name,
                'year': future_year,
                'target': target_key,
                'target_name': target_info['name'],
                'unit': target_info['unit'],
                'predicted': pred,
                'ci_lower': pred - ci_factor,
                'ci_upper': pred + ci_factor,
                'horizon': h,
            })

    return pd.DataFrame(projections)


def predict():
    """Pipeline de prédiction complet multi-cibles."""
    print("=" * 70)
    print("  PRÉDICTIONS & PROJECTIONS — Togo & Zone UEMOA")
    print("=" * 70)

    df = load_processed_data()
    all_historical = []
    all_projections = []

    for target_key, target_info in PREDICTION_TARGETS.items():
        model_path = os.path.join(MODELS_DIR, f'model_{target_key}.joblib')
        if not os.path.exists(model_path):
            print(f"\n  ⚠ Modèle manquant pour '{target_key}' — saut")
            continue

        print(f"\n  ▸ {target_info['name']}")

        # 1. Prédictions historiques
        hist = predict_historical(df, target_key)
        all_historical.append(hist)

        # Focus Togo
        togo_hist = hist[hist['country_code'] == FOCUS_COUNTRY]
        if not togo_hist.empty:
            mae = togo_hist['error'].abs().mean()
            print(f"    Togo — MAE historique : {mae:.2f} {target_info['unit']}")

        # 2. Projections futures
        proj = project_future(df, target_key)
        all_projections.append(proj)

        # Focus Togo projections
        togo_proj = proj[proj['country_code'] == FOCUS_COUNTRY]
        if not togo_proj.empty:
            print(f"    Togo — Projections {int(togo_proj['year'].min())}-"
                  f"{int(togo_proj['year'].max())} :")
            for _, row in togo_proj.iterrows():
                print(f"      {int(row['year'])} : {row['predicted']:.1f} "
                      f"[{row['ci_lower']:.1f} — {row['ci_upper']:.1f}] "
                      f"{target_info['unit']}")

    # Sauvegarder
    if all_historical:
        hist_df = pd.concat(all_historical, ignore_index=True)
        hist_path = os.path.join(PREDICTIONS_DIR, 'predictions.csv')
        hist_df.to_csv(hist_path, index=False, encoding='utf-8')
        print(f"\n  ✓ Prédictions historiques : {hist_path}")

    if all_projections:
        proj_df = pd.concat(all_projections, ignore_index=True)
        proj_path = os.path.join(PREDICTIONS_DIR, 'projections.csv')
        proj_df.to_csv(proj_path, index=False, encoding='utf-8')
        print(f"  ✓ Projections futures    : {proj_path}")

    # Analyse d'impact énergie ↔ santé
    print(f"\n  ── Analyse d'Impact Énergie ↔ Développement (Togo) ──")
    togo = df[df['country_code'] == FOCUS_COUNTRY].sort_values('year')
    if not togo.empty:
        if 'EG.ELC.ACCS.ZS' in togo.columns and 'SH.DYN.MORT' in togo.columns:
            first = togo.iloc[0]
            last = togo.iloc[-1]
            acc_change = last.get('EG.ELC.ACCS.ZS', 0) - first.get('EG.ELC.ACCS.ZS', 0)
            mort_change = last.get('SH.DYN.MORT', 0) - first.get('SH.DYN.MORT', 0)
            print(f"    Accès électricité : {first.get('EG.ELC.ACCS.ZS', 0):.1f}% → "
                  f"{last.get('EG.ELC.ACCS.ZS', 0):.1f}% "
                  f"({'↑' if acc_change > 0 else '↓'} {abs(acc_change):.1f} pts)")
            print(f"    Mortalité infant. : {first.get('SH.DYN.MORT', 0):.1f} → "
                  f"{last.get('SH.DYN.MORT', 0):.1f} "
                  f"({'↓' if mort_change < 0 else '↑'} {abs(mort_change):.1f})")
            if acc_change != 0:
                ratio = mort_change / acc_change
                print(f"    Impact estimé : {abs(ratio):.2f} points de mortalité "
                      f"par point d'accès électrique")

    print(f"\n  ✓ Prédictions terminées.")
    return hist_df if all_historical else None


if __name__ == '__main__':
    predict()
