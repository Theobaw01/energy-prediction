"""
ETL — Extract : Collecte des données énergétiques et de développement (Togo + UEMOA).
Source : API Banque Mondiale (World Development Indicators)

Contexte BCEAO : La surveillance multilatérale de la zone UEMOA nécessite
des données fiables sur les indicateurs de développement. L'énergie est un
proxy clé de l'activité économique et du bien-être des populations.
"""
import os
import json
import time
import urllib.request
import urllib.error
import pandas as pd
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import (
    COUNTRIES, INDICATORS, INDICATOR_GROUPS,
    START_YEAR, END_YEAR, RAW_DIR, FOCUS_COUNTRY_NAME
)

# ── Paramètres API ───────────────────────────────────────────────────────────
API_BASE = "https://api.worldbank.org/v2"
MAX_RETRIES = 3
RETRY_DELAY = 2
PER_PAGE = 5000
USER_AGENT = 'BCEAO-EnergyAnalytics/1.0'

# Mapping noms anglais API → noms français
COUNTRY_NAME_FR = {
    'Togo': 'Togo',
    'Senegal': 'Sénégal',
    "Cote d'Ivoire": "Côte d'Ivoire",
    'Benin': 'Bénin',
    'Burkina Faso': 'Burkina Faso',
    'Mali': 'Mali',
    'Niger': 'Niger',
    'Guinea-Bissau': 'Guinée-Bissau',
}


def fetch_indicator(indicator_code: str, country_codes: list[str],
                    start_year: int, end_year: int) -> pd.DataFrame:
    """
    Récupère un indicateur depuis l'API Banque Mondiale avec retry automatique.
    Filtre les valeurs nulles à la source pour un dataset propre.
    """
    countries = ';'.join(country_codes)
    url = (
        f"{API_BASE}/country/{countries}/indicator/{indicator_code}"
        f"?date={start_year}:{end_year}&format=json&per_page={PER_PAGE}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"    ✗ Échec pour {indicator_code}: {e}")
                return pd.DataFrame()

    if not isinstance(data, list) or len(data) < 2 or data[1] is None:
        return pd.DataFrame()

    records = []
    for entry in data[1]:
        if entry.get('value') is not None:
            en_name = entry['country']['value']
            records.append({
                'country_code': entry['country']['id'],
                'country_name': COUNTRY_NAME_FR.get(en_name, en_name),
                'year': int(entry['date']),
                'indicator_code': indicator_code,
                'indicator_name': entry['indicator']['value'],
                'value': float(entry['value']),
            })

    return pd.DataFrame(records)


def extract_all() -> pd.DataFrame:
    """
    Pipeline d'extraction complet :
    - Collecte tous les indicateurs pour le Togo + pays UEMOA
    - Rapport de couverture détaillé
    - Sauvegarde CSV avec métadonnées
    """
    print("=" * 70)
    print(f"  EXTRACTION — Données {FOCUS_COUNTRY_NAME} & Zone UEMOA")
    print(f"  Cadre : Analyse énergétique & développement (objectifs BCEAO)")
    print("=" * 70)
    print(f"\n  Pays         : {len(COUNTRIES)} (focus {FOCUS_COUNTRY_NAME})")
    print(f"  Période      : {START_YEAR} — {END_YEAR}")
    print(f"  Indicateurs  : {len(INDICATORS)}")
    print(f"  Horodatage   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    country_codes = list(COUNTRIES.keys())
    all_data = []
    stats = {'success': 0, 'empty': 0}

    for group_name, group_indicators in INDICATOR_GROUPS.items():
        print(f"\n  ▸ {group_name} ({len(group_indicators)} indicateurs)")

        for code, name in group_indicators.items():
            df = fetch_indicator(code, country_codes, START_YEAR, END_YEAR)

            if df.empty:
                stats['empty'] += 1
                print(f"    ○ {name}: aucune donnée")
            else:
                all_data.append(df)
                n_c = df['country_code'].nunique()
                yr = f"{df['year'].min()}-{df['year'].max()}"
                stats['success'] += 1
                print(f"    ● {name}: {len(df)} obs. ({n_c} pays, {yr})")

    if not all_data:
        print("\n  ✗ Aucune donnée extraite.")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['country_code', 'year', 'indicator_code'])
    combined = combined.reset_index(drop=True)
    combined['extraction_date'] = datetime.now().strftime('%Y-%m-%d')
    combined['source'] = 'World Bank WDI'

    output_path = os.path.join(RAW_DIR, 'energy_data_raw.csv')
    combined.to_csv(output_path, index=False, encoding='utf-8')

    # Rapport
    print("\n" + "=" * 70)
    print("  RAPPORT D'EXTRACTION")
    print("=" * 70)
    print(f"  Enregistrements totaux : {len(combined):,}")
    print(f"  Indicateurs OK         : {stats['success']}/{len(INDICATORS)}")
    print(f"  Fichier                : {output_path}")

    print(f"\n  Couverture par pays :")
    for code, name in COUNTRIES.items():
        cd = combined[combined['country_code'] == code]
        ni = cd['indicator_code'].nunique()
        cov = ni / len(INDICATORS) * 100
        tag = " ◀ FOCUS" if code == 'TG' else ""
        print(f"    {name:20s} : {ni:2d}/{len(INDICATORS)} indicateurs "
              f"({cov:.0f}%) — {len(cd)} obs.{tag}")

    print(f"\n  ✓ Extraction terminée.")
    return combined


if __name__ == '__main__':
    extract_all()
