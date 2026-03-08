"""
ETL — Extract : Collecte des données énergétiques via l'API Banque Mondiale.
Source : https://api.worldbank.org/v2/
"""
import os
import json
import urllib.request
import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import COUNTRIES, INDICATORS, START_YEAR, END_YEAR, RAW_DIR


def fetch_indicator(indicator_code: str, country_codes: list[str],
                    start_year: int, end_year: int) -> pd.DataFrame:
    """
    Récupère un indicateur depuis l'API Banque Mondiale pour une liste de pays.
    """
    countries = ';'.join(country_codes)
    url = (
        f"https://api.worldbank.org/v2/country/{countries}/indicator/{indicator_code}"
        f"?date={start_year}:{end_year}&format=json&per_page=1000"
    )

    print(f"  Fetching {indicator_code}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"  ⚠ Erreur pour {indicator_code}: {e}")
        return pd.DataFrame()

    if len(data) < 2 or data[1] is None:
        print(f"  ⚠ Pas de données pour {indicator_code}")
        return pd.DataFrame()

    records = []
    for entry in data[1]:
        records.append({
            'country_code': entry['country']['id'],
            'country_name': entry['country']['value'],
            'year': int(entry['date']),
            'indicator_code': indicator_code,
            'indicator_name': entry['indicator']['value'],
            'value': entry['value'],
        })

    return pd.DataFrame(records)


def extract_all():
    """
    Extrait toutes les données et les sauvegarde en CSV.
    """
    print("=" * 60)
    print("EXTRACTION DES DONNÉES BANQUE MONDIALE")
    print(f"Pays : {', '.join(COUNTRIES.values())}")
    print(f"Période : {START_YEAR} - {END_YEAR}")
    print(f"Indicateurs : {len(INDICATORS)}")
    print("=" * 60)

    country_codes = list(COUNTRIES.keys())
    all_data = []

    for code, name in INDICATORS.items():
        df = fetch_indicator(code, country_codes, START_YEAR, END_YEAR)
        if not df.empty:
            all_data.append(df)
            print(f"  ✓ {name}: {len(df)} enregistrements")

    if not all_data:
        print("❌ Aucune donnée extraite.")
        return

    # Combiner toutes les données
    combined = pd.concat(all_data, ignore_index=True)

    # Sauvegarder
    output_path = os.path.join(RAW_DIR, 'energy_data_raw.csv')
    combined.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\n✅ Extraction terminée : {len(combined)} enregistrements")
    print(f"   Fichier : {output_path}")
    print(f"   Pays : {combined['country_code'].nunique()}")
    print(f"   Indicateurs : {combined['indicator_code'].nunique()}")
    print(f"   Années : {combined['year'].min()} - {combined['year'].max()}")

    return combined


if __name__ == '__main__':
    extract_all()
