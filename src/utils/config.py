"""
Configuration du projet de prédiction énergétique — Gabon.
"""
import os

# Chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Créer les dossiers si nécessaire
for d in [RAW_DIR, PROCESSED_DIR, PREDICTIONS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Pays : Gabon (+ pays voisins pour comparaison)
COUNTRIES = {
    'GAB': 'Gabon',
    'CMR': 'Cameroun',
    'COG': 'Congo',
    'GNQ': 'Guinée Équatoriale',
}

# Indicateurs Banque Mondiale - Énergie & Économie
INDICATORS = {
    'EG.USE.ELEC.KH.PC': 'Consommation électrique (kWh/hab)',
    'EG.USE.PCAP.KG.OE': 'Consommation énergie (kg pétrole/hab)',
    'EG.ELC.ACCS.ZS': "Accès à l'électricité (%)",
    'EG.ELC.ACCS.UR.ZS': "Accès électricité urbain (%)",
    'EG.ELC.ACCS.RU.ZS': "Accès électricité rural (%)",
    'EG.FEC.RNEW.ZS': 'Énergie renouvelable (%)',
    'EG.ELC.LOSS.ZS': 'Pertes électriques (%)',
    'NY.GDP.PCAP.CD': 'PIB par habitant (USD)',
    'NY.GDP.MKTP.KD.ZG': 'Croissance PIB (%)',
    'SP.POP.TOTL': 'Population totale',
    'SP.URB.TOTL.IN.ZS': 'Population urbaine (%)',
    'SP.POP.GROW': 'Croissance population (%)',
    'FP.CPI.TOTL.ZG': 'Inflation (%)',
}

# Paramètres
START_YEAR = 2000
END_YEAR = 2023
TARGET_INDICATOR = 'EG.USE.ELEC.KH.PC'
FOCUS_COUNTRY = 'GAB'
RANDOM_STATE = 42
TEST_SIZE = 0.2
