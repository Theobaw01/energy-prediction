"""
Dashboard BI interactif — Prédiction Énergétique Zone UEMOA
Streamlit Application
"""
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Config Streamlit
st.set_page_config(
    page_title="Énergie Gabon — Dashboard BI",
    page_icon="⚡",
    layout="wide",
)

# Chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
PREDICTIONS_DIR = os.path.join(BASE_DIR, 'data', 'predictions')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


@st.cache_data
def load_data():
    """Charge les données traitées et les prédictions."""
    processed_path = os.path.join(PROCESSED_DIR, 'energy_data_processed.csv')
    predictions_path = os.path.join(PREDICTIONS_DIR, 'predictions.csv')

    data = {}
    if os.path.exists(processed_path):
        data['processed'] = pd.read_csv(processed_path)
    if os.path.exists(predictions_path):
        data['predictions'] = pd.read_csv(predictions_path)
    if os.path.exists(os.path.join(MODELS_DIR, 'results.csv')):
        data['results'] = pd.read_csv(os.path.join(MODELS_DIR, 'results.csv'))

    return data


def render_kpis(df):
    """Affiche les KPIs principaux."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Pays analysés",
            df['country_code'].nunique(),
            help="Nombre de pays de la zone UEMOA"
        )
    with col2:
        years = f"{df['year'].min()} - {df['year'].max()}"
        st.metric("Période", years)
    with col3:
        if 'EG.USE.ELEC.KH.PC' in df.columns:
            avg = df['EG.USE.ELEC.KH.PC'].mean()
            st.metric("Conso. moy.", f"{avg:.0f} kWh/hab")
    with col4:
        if 'EG.ELC.ACCS.ZS' in df.columns:
            access = df[df['year'] == df['year'].max()]['EG.ELC.ACCS.ZS'].mean()
            st.metric("Accès électricité", f"{access:.1f}%")


def plot_consumption_trends(df):
    """Graphique des tendances de consommation par pays."""
    if 'EG.USE.ELEC.KH.PC' not in df.columns:
        st.warning("Données de consommation non disponibles")
        return

    fig = px.line(
        df, x='year', y='EG.USE.ELEC.KH.PC',
        color='country_name',
        title='Consommation Électrique par Habitant — Gabon & Région',
        labels={
            'year': 'Année',
            'EG.USE.ELEC.KH.PC': 'Consommation (kWh/hab)',
            'country_name': 'Pays'
        },
        template='plotly_dark',
    )
    fig.update_layout(height=450, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


def plot_access_map(df):
    """Graphique d'accès à l'électricité."""
    if 'EG.ELC.ACCS.ZS' not in df.columns:
        return

    latest = df[df['year'] == df['year'].max()]
    fig = px.bar(
        latest.sort_values('EG.ELC.ACCS.ZS', ascending=True),
        x='EG.ELC.ACCS.ZS', y='country_name',
        orientation='h',
        title=f"Accès à l'Électricité par Pays ({df['year'].max()})",
        labels={
            'EG.ELC.ACCS.ZS': 'Accès (%)',
            'country_name': ''
        },
        color='EG.ELC.ACCS.ZS',
        color_continuous_scale='YlOrRd',
        template='plotly_dark',
    )
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def plot_predictions(predictions_df):
    """Graphique prédictions vs réel."""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=['Prédictions vs Valeurs Réelles']
    )

    for country in predictions_df['country_name'].unique():
        country_data = predictions_df[predictions_df['country_name'] == country]

        fig.add_trace(go.Scatter(
            x=country_data['year'], y=country_data['actual'],
            name=f'{country} (réel)',
            mode='lines+markers',
            line=dict(width=2),
        ))
        fig.add_trace(go.Scatter(
            x=country_data['year'], y=country_data['predicted'],
            name=f'{country} (prédit)',
            mode='lines',
            line=dict(dash='dash', width=1.5),
        ))

    fig.update_layout(
        title='Prédictions vs Valeurs Réelles — Consommation Électrique',
        xaxis_title='Année',
        yaxis_title='kWh/habitant',
        template='plotly_dark',
        height=500,
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_model_comparison(results_df):
    """Comparaison des performances des modèles."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['RMSE (↓)', 'MAE (↓)', 'R² (↑)']
    )

    colors = px.colors.qualitative.Set2

    for i, metric in enumerate(['rmse', 'mae', 'r2']):
        fig.add_trace(go.Bar(
            x=results_df['model'],
            y=results_df[metric],
            marker_color=colors[:len(results_df)],
            showlegend=False,
        ), row=1, col=i+1)

    fig.update_layout(
        title='Comparaison des Modèles',
        template='plotly_dark',
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_energy_mix(df):
    """Graphique mix énergétique (renouvelable)."""
    if 'EG.FEC.RNEW.ZS' not in df.columns:
        return

    fig = px.area(
        df, x='year', y='EG.FEC.RNEW.ZS',
        color='country_name',
        title='Part des Énergies Renouvelables — Gabon & Région',
        labels={
            'year': 'Année',
            'EG.FEC.RNEW.ZS': 'Renouvelable (%)',
            'country_name': 'Pays'
        },
        template='plotly_dark',
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def main():
    # Header
    st.title("⚡ Prédiction Énergétique — Gabon")
    st.markdown(
        "Dashboard BI interactif pour l'analyse et la prédiction "
        "de la consommation énergétique au Gabon (Libreville et régions)."
    )
    st.divider()

    # Charger les données
    data = load_data()

    if 'processed' not in data:
        st.error("⚠ Données non trouvées. Exécutez le pipeline ETL d'abord.")
        st.code("python src/etl/extract.py\npython src/etl/transform.py\npython src/models/train.py")
        return

    df = data['processed']

    # Sidebar — Filtres
    st.sidebar.header("🎛️ Filtres")
    countries = st.sidebar.multiselect(
        "Pays",
        options=sorted(df['country_name'].unique()),
        default=sorted(df['country_name'].unique()),
    )
    year_range = st.sidebar.slider(
        "Période",
        int(df['year'].min()), int(df['year'].max()),
        (int(df['year'].min()), int(df['year'].max()))
    )

    # Filtrer
    filtered = df[
        (df['country_name'].isin(countries)) &
        (df['year'].between(*year_range))
    ]

    # KPIs
    render_kpis(filtered)
    st.divider()

    # Onglets
    tab1, tab2, tab3 = st.tabs(["📊 Tendances", "🤖 Prédictions", "🔋 Mix Énergétique"])

    with tab1:
        plot_consumption_trends(filtered)
        col1, col2 = st.columns(2)
        with col1:
            plot_access_map(filtered)
        with col2:
            # GDP vs consommation
            if 'NY.GDP.PCAP.CD' in filtered.columns and 'EG.USE.ELEC.KH.PC' in filtered.columns:
                latest = filtered[filtered['year'] == filtered['year'].max()]
                fig = px.scatter(
                    latest, x='NY.GDP.PCAP.CD', y='EG.USE.ELEC.KH.PC',
                    size='SP.POP.TOTL' if 'SP.POP.TOTL' in latest.columns else None,
                    color='country_name',
                    title='PIB vs Consommation Électrique',
                    labels={
                        'NY.GDP.PCAP.CD': 'PIB/hab (USD)',
                        'EG.USE.ELEC.KH.PC': 'kWh/hab',
                        'country_name': 'Pays'
                    },
                    template='plotly_dark',
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if 'predictions' in data:
            pred_filtered = data['predictions'][
                data['predictions']['country_name'].isin(countries)
            ]
            plot_predictions(pred_filtered)

            if 'results' in data:
                plot_model_comparison(data['results'])
        else:
            st.info("Exécutez le modèle pour voir les prédictions : `python src/models/train.py`")

    with tab3:
        plot_energy_mix(filtered)

    # Footer
    st.divider()
    st.caption("Données : Banque Mondiale Open Data | Modèles : Scikit-learn, XGBoost, LightGBM")


if __name__ == '__main__':
    main()
