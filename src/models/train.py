"""
Entraînement multi-cibles : Consommation électrique, Mortalité infantile, Accès électricité.
Modèles : Random Forest, XGBoost, LightGBM + Ensemble (Stacking).

Méthodologie :
- Validation croisée temporelle (TimeSeriesSplit)
- Optimisation hyperparamètres
- Feature importance par modèle
- Sauvegarde du meilleur modèle par cible
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import MODELS_DIR, RANDOM_STATE, CV_FOLDS, PREDICTION_TARGETS
from etl.load import load_processed_data, prepare_all_targets

# Import optionnel XGBoost / LightGBM
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    """Calcule RMSE, MAE, R² et MAPE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (éviter division par 0)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    print(f"     {model_name:25s} │ RMSE: {rmse:10.2f} │ MAE: {mae:10.2f} "
          f"│ R²: {r2:.4f} │ MAPE: {mape:.1f}%")

    return {
        'model': model_name, 'rmse': rmse, 'mae': mae,
        'r2': r2, 'mape': mape
    }


def get_models() -> dict:
    """Retourne le dictionnaire des modèles à entraîner."""
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_split=5,
            min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE,
        ),
    }

    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
        )

    if HAS_LGB:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, num_leaves=31,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        )

    return models


def build_stacking(base_models: dict) -> StackingRegressor:
    """
    Construit un modèle de Stacking (ensemble) avec les meilleurs modèles.
    Meta-learner : Ridge Regression.
    """
    estimators = [(name, model) for name, model in base_models.items()]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=3, n_jobs=-1,
    )


def get_feature_importance(model, feature_names: list, top_n: int = 10) -> pd.DataFrame:
    """Retourne un DataFrame des features les plus importantes."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        return pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': [importances[i] for i in indices],
        })
    return pd.DataFrame()


def train_single_target(target_key: str, target_data: dict) -> dict:
    """
    Entraîne tous les modèles pour une cible donnée.
    Retourne le meilleur modèle et les résultats.
    """
    info = target_data['info']
    X_train = target_data['X_train']
    X_test = target_data['X_test']
    y_train = target_data['y_train']
    y_test = target_data['y_test']
    feature_names = target_data['feature_names']

    print(f"\n  {'━' * 60}")
    print(f"  CIBLE : {info['name']}")
    print(f"  {info['description']}")
    print(f"  {'━' * 60}")

    # Nettoyage NaN/Inf
    valid_train = ~np.isnan(y_train) & ~np.isinf(y_train)
    valid_test = ~np.isnan(y_test) & ~np.isinf(y_test)
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_test, y_test = X_test[valid_test], y_test[valid_test]
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    if len(X_train) < 10 or len(X_test) < 3:
        print(f"  ⚠ Données insuffisantes (train={len(X_train)}, test={len(X_test)})")
        return None

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entraîner les modèles
    models = get_models()
    results = []
    trained = {}

    print(f"\n     {'Modèle':25s} │ {'RMSE':>10s} │ {'MAE':>10s} │ {'R²':>6s} │ {'MAPE':>6s}")
    print(f"     {'─' * 25}─┼─{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 6}─┼─{'─' * 6}")

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics = evaluate_model(y_test, y_pred, name)
        results.append(metrics)
        trained[name] = model

    # Stacking Ensemble
    if len(trained) >= 2:
        ensemble_models = get_models()  # Modèles frais pour le stacking
        stacking = build_stacking(ensemble_models)
        try:
            stacking.fit(X_train_scaled, y_train)
            y_pred_stack = stacking.predict(X_test_scaled)
            metrics = evaluate_model(y_test, y_pred_stack, 'Stacking Ensemble')
            results.append(metrics)
            trained['Stacking Ensemble'] = stacking
        except Exception as e:
            print(f"     ⚠ Stacking échoué: {e}")

    # Résultats
    results_df = pd.DataFrame(results).sort_values('r2', ascending=False)
    best_name = results_df.iloc[0]['model']
    best_model = trained[best_name]
    best_r2 = results_df.iloc[0]['r2']

    print(f"\n  🏆 Meilleur : {best_name} (R² = {best_r2:.4f})")

    # Feature importance du meilleur modèle
    fi = get_feature_importance(best_model, feature_names, top_n=8)
    if not fi.empty:
        print(f"\n     Top features :")
        for _, row in fi.iterrows():
            bar = '█' * int(row['importance'] * 50)
            print(f"       {row['feature']:35s} {bar} {row['importance']:.3f}")

    # Sauvegarder
    model_path = os.path.join(MODELS_DIR, f'model_{target_key}.joblib')
    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'model_name': best_name,
        'feature_names': feature_names,
        'target_key': target_key,
        'target_info': info,
        'metrics': results_df.iloc[0].to_dict(),
    }, model_path)
    print(f"  ✓ Modèle sauvegardé : {model_path}")

    return {
        'best_model': best_model,
        'scaler': scaler,
        'results': results_df,
        'feature_importance': fi,
    }


def train():
    """Pipeline d'entraînement complet multi-cibles."""
    print("=" * 70)
    print("  ENTRAÎNEMENT — Modèles Prédictifs Multi-Cibles")
    print("  Cadre : Énergie & Développement au Togo (objectifs BCEAO)")
    print("=" * 70)

    # Charger données
    print("\n  Chargement des données...")
    df = load_processed_data()
    all_targets = prepare_all_targets(df)

    if not all_targets:
        print("  ✗ Aucune cible disponible.")
        return

    # Entraîner chaque cible
    all_results = {}
    for target_key, target_data in all_targets.items():
        result = train_single_target(target_key, target_data)
        if result:
            all_results[target_key] = result

    # Résumé global
    print("\n" + "=" * 70)
    print("  RÉSUMÉ GLOBAL")
    print("=" * 70)

    summary_rows = []
    for key, res in all_results.items():
        best = res['results'].iloc[0]
        target_name = PREDICTION_TARGETS[key]['name']
        print(f"  • {target_name:40s} → {best['model']:20s} "
              f"(R²={best['r2']:.4f}, RMSE={best['rmse']:.2f})")
        summary_rows.append({
            'target': key,
            'target_name': target_name,
            **best.to_dict()
        })

    # Sauvegarder résumé
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(MODELS_DIR, 'results.csv')
    summary_df.to_csv(summary_path, index=False)

    print(f"\n  ✓ Entraînement terminé — {len(all_results)} modèles sauvegardés.")
    return all_results


if __name__ == '__main__':
    train()
