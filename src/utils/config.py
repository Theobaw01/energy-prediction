"""
Configuration — Analyse Prédictive Énergétique & Développement au Togo
Cadre : Objectifs de développement économique — Zone UEMOA / BCEAO

Le Togo, membre de l'UEMOA, fait face à un défi énergétique majeur :
  - Accès électricité : 59% (urbain 95% vs rural 30%)
  - Consommation : ~192 kWh/hab (vs ~6 000 kWh/hab en France)
  - Forte dépendance aux énergies renouvelables traditionnelles (75%)

Ce projet analyse les liens entre énergie, économie et développement humain
au Togo pour produire des modèles prédictifs utiles à la prise de décision
dans le cadre de la surveillance multilatérale de la BCEAO.
"""
import os

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')
FIGURES_DIR = os.path.join(DATA_DIR, 'figures')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

for d in [RAW_DIR, PROCESSED_DIR, PREDICTIONS_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Pays ─────────────────────────────────────────────────────────────────────
# Focus : TOGO — avec les autres pays UEMOA pour benchmark régional BCEAO
FOCUS_COUNTRY = 'TG'
FOCUS_COUNTRY_NAME = 'Togo'

# Codes ISO2 (format API Banque Mondiale)
COUNTRIES = {
    # --- Focus principal ---
    'TG': 'Togo',
    # --- Autres pays UEMOA (zone BCEAO) pour comparaison ---
    'SN': 'Sénégal',
    'CI': "Côte d'Ivoire",
    'BJ': 'Bénin',
    'BF': 'Burkina Faso',
    'ML': 'Mali',
    'NE': 'Niger',
    'GW': 'Guinée-Bissau',
}

# Mapping ISO3 → ISO2 (pour l'API qui accepte les deux mais renvoie ISO2)
ISO3_TO_ISO2 = {
    'TGO': 'TG', 'SEN': 'SN', 'CIV': 'CI', 'BEN': 'BJ',
    'BFA': 'BF', 'MLI': 'ML', 'NER': 'NE', 'GNB': 'GW',
}

# ── Indicateurs Banque Mondiale ──────────────────────────────────────────────
# Organisés par domaine d'impact, alignés sur les axes BCEAO + développement

# 1. ÉNERGIE — Cœur du projet : consommation et accès
ENERGY_INDICATORS = {
    'EG.USE.ELEC.KH.PC': 'Consommation électrique (kWh/hab)',
    'EG.USE.PCAP.KG.OE': 'Consommation énergie primaire (kg pétrole/hab)',
    'EG.ELC.ACCS.ZS': "Taux d'accès à l'électricité (%)",
    'EG.ELC.ACCS.UR.ZS': "Accès électricité — urbain (%)",
    'EG.ELC.ACCS.RU.ZS': "Accès électricité — rural (%)",
    'EG.FEC.RNEW.ZS': 'Part énergies renouvelables (%)',
}

# 2. ÉCONOMIE & CROISSANCE — Suivi macroéconomique BCEAO
MACRO_INDICATORS = {
    'NY.GDP.PCAP.CD': 'PIB par habitant (USD)',
    'NY.GDP.MKTP.KD.ZG': 'Croissance du PIB réel (%)',
    'FP.CPI.TOTL.ZG': 'Inflation — prix consommation (%)',
    'FM.LBL.BMNY.GD.ZS': 'Masse monétaire M2 (% PIB)',
    'NE.EXP.GNFS.ZS': 'Exportations (% PIB)',
    'NE.IMP.GNFS.ZS': 'Importations (% PIB)',
    'BX.TRF.PWKR.DT.GD.ZS': 'Transferts de fonds reçus (% PIB)',
}

# 3. SANTÉ & DÉVELOPPEMENT HUMAIN — Impact social de l'énergie
HEALTH_INDICATORS = {
    'SH.XPD.CHEX.PC.CD': 'Dépenses de santé par habitant (USD)',
    'SH.DYN.MORT': 'Mortalité infantile (pour 1 000 naissances)',
    'SH.STA.MMRT': 'Mortalité maternelle (pour 100 000)',
    'SP.DYN.LE00.IN': 'Espérance de vie à la naissance (années)',
    'SH.H2O.BASW.ZS': 'Accès eau potable de base (%)',
    'SH.STA.BASS.ZS': 'Accès assainissement de base (%)',
}

# 4. DÉMOGRAPHIE & INFRASTRUCTURE
DEMO_INDICATORS = {
    'SP.POP.TOTL': 'Population totale',
    'SP.POP.GROW': 'Croissance démographique (%)',
    'SP.URB.TOTL.IN.ZS': "Taux d'urbanisation (%)",
    'SL.UEM.TOTL.ZS': 'Taux de chômage (%)',
    'IT.CEL.SETS.P2': 'Abonnements téléphone mobile (pour 100 hab)',
    'IT.NET.USER.ZS': 'Utilisateurs internet (%)',
}

# Regroupement complet
INDICATORS = {}
INDICATORS.update(ENERGY_INDICATORS)
INDICATORS.update(MACRO_INDICATORS)
INDICATORS.update(HEALTH_INDICATORS)
INDICATORS.update(DEMO_INDICATORS)

# Groupes pour le dashboard et l'analyse
INDICATOR_GROUPS = {
    'Énergie & Électricité': ENERGY_INDICATORS,
    'Économie & Croissance': MACRO_INDICATORS,
    'Santé & Développement': HEALTH_INDICATORS,
    'Démographie & Infrastructure': DEMO_INDICATORS,
}

# Labels courts pour les graphiques
INDICATOR_SHORT = {
    'EG.USE.ELEC.KH.PC': 'Conso. élec.',
    'EG.USE.PCAP.KG.OE': 'Énergie prim.',
    'EG.ELC.ACCS.ZS': 'Accès élec.',
    'EG.ELC.ACCS.UR.ZS': 'Accès urb.',
    'EG.ELC.ACCS.RU.ZS': 'Accès rur.',
    'EG.FEC.RNEW.ZS': 'Renouvelable',
    'NY.GDP.PCAP.CD': 'PIB/hab',
    'NY.GDP.MKTP.KD.ZG': 'Croiss. PIB',
    'FP.CPI.TOTL.ZG': 'Inflation',
    'FM.LBL.BMNY.GD.ZS': 'M2/PIB',
    'SH.DYN.MORT': 'Mort. infant.',
    'SH.STA.MMRT': 'Mort. mater.',
    'SP.DYN.LE00.IN': 'Esp. vie',
    'SP.POP.TOTL': 'Population',
}

# ── Cibles de prédiction (3 axes stratégiques) ──────────────────────────────
PREDICTION_TARGETS = {
    'energy': {
        'indicator': 'EG.USE.ELEC.KH.PC',
        'name': 'Consommation électrique (kWh/hab)',
        'unit': 'kWh/hab',
        'description': 'Prédiction de la consommation électrique par habitant',
    },
    'health': {
        'indicator': 'SH.DYN.MORT',
        'name': 'Mortalité infantile',
        'unit': 'pour 1 000',
        'description': "Prédiction de la mortalité infantile (lien énergie-santé)",
    },
    'access': {
        'indicator': 'EG.ELC.ACCS.ZS',
        'name': "Taux d'accès à l'électricité",
        'unit': '%',
        'description': "Projection du taux d'électrification national",
    },
}

TARGET_INDICATOR = 'EG.USE.ELEC.KH.PC'

# ── Paramètres de modélisation ───────────────────────────────────────────────
START_YEAR = 2000
END_YEAR = 2023
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
FORECAST_HORIZON = 5  # Projections 2024-2028

# ── Thème visuel ─────────────────────────────────────────────────────────────
COLORS = {
    'primary': '#1B4F72',       # Bleu BCEAO
    'secondary': '#2E86C1',
    'accent': '#F39C12',        # Or
    'success': '#27AE60',
    'danger': '#E74C3C',
    'warning': '#E67E22',
    'dark': '#2C3E50',
    'light': '#ECF0F1',
    'togo_green': '#006A4E',    # Vert Togo
    'togo_yellow': '#FFCE00',   # Jaune Togo
    'togo_red': '#D21034',      # Rouge Togo
    'uemoa_blue': '#003399',    # Bleu UEMOA
}
