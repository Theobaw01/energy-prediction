"""
Dashboard BI — Energie et Developpement au Togo
Zone UEMOA · Cadre analytique BCEAO
Streamlit · Plotly · scikit-learn · XGBoost · LightGBM
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BI Energie — Togo | BCEAO",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "energy_data_processed.csv")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "predictions", "predictions.csv")
PROJECTIONS_PATH = os.path.join(BASE_DIR, "data", "predictions", "projections.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "models", "results.csv")

# Palette sobre
PAL = {
    "navy": "#1B3A4B",
    "blue": "#2E6F8E",
    "teal": "#3A9E92",
    "gold": "#C9A227",
    "green": "#3D7A5F",
    "red": "#B03A2E",
    "orange": "#D4780A",
    "slate": "#5D6D7E",
    "light": "#ECF0F1",
    "muted": "#95A5A6",
    "bg": "#0E1117",
}

COUNTRY_PAL = {
    "Togo": "#1B6B50",
    "Senegal": "#2E6F8E",
    "Sénégal": "#2E6F8E",
    "Côte d'Ivoire": "#C9A227",
    "Bénin": "#B03A2E",
    "Burkina Faso": "#7D3C98",
    "Mali": "#3D7A5F",
    "Niger": "#D4780A",
    "Guinée-Bissau": "#1ABC9C",
}

TMPL = "plotly_dark"

# Metadata des indicateurs
IND = {
    "EG.USE.ELEC.KH.PC":     ("Consommation electrique",  "kWh/hab",    "Energie",     True),
    "EG.USE.PCAP.KG.OE":     ("Energie primaire",         "kg pet/hab", "Energie",     False),
    "EG.ELC.ACCS.ZS":        ("Acces electricite",        "%",          "Energie",     True),
    "EG.ELC.ACCS.UR.ZS":     ("Acces electr. urbain",     "%",          "Energie",     True),
    "EG.ELC.ACCS.RU.ZS":     ("Acces electr. rural",      "%",          "Energie",     True),
    "EG.FEC.RNEW.ZS":        ("Energies renouvelables",   "%",          "Energie",     True),
    "NY.GDP.PCAP.CD":        ("PIB par habitant",          "USD",        "Economie",    True),
    "NY.GDP.MKTP.KD.ZG":     ("Croissance du PIB",        "%",          "Economie",    True),
    "FP.CPI.TOTL.ZG":        ("Inflation",                "%",          "Economie",    False),
    "FM.LBL.BMNY.GD.ZS":     ("Masse monetaire M2/PIB",   "%",          "Economie",    False),
    "NE.EXP.GNFS.ZS":        ("Exportations / PIB",       "%",          "Economie",    True),
    "NE.IMP.GNFS.ZS":        ("Importations / PIB",       "%",          "Economie",    False),
    "BX.TRF.PWKR.DT.GD.ZS":  ("Transferts de fonds",     "% PIB",      "Economie",    True),
    "SH.DYN.MORT":           ("Mortalite infantile",       "p. 1000",    "Sante",       False),
    "SH.STA.MMRT":           ("Mortalite maternelle",      "/100k",      "Sante",       False),
    "SP.DYN.LE00.IN":        ("Esperance de vie",          "ans",        "Sante",       True),
    "SH.XPD.CHEX.PC.CD":     ("Depenses de sante",        "USD/hab",    "Sante",       True),
    "SH.H2O.BASW.ZS":       ("Acces eau potable",         "%",          "Sante",       True),
    "SH.STA.BASS.ZS":       ("Assainissement",            "%",          "Sante",       True),
    "SP.POP.TOTL":           ("Population",                "",           "Demographie", False),
    "SP.POP.GROW":           ("Croissance demogr.",        "%",          "Demographie", False),
    "SP.URB.TOTL.IN.ZS":    ("Urbanisation",              "%",          "Demographie", False),
    "SL.UEM.TOTL.ZS":       ("Chomage",                   "%",          "Demographie", False),
    "IT.CEL.SETS.P2":        ("Abonnes mobile / 100",     "",           "Demographie", True),
    "IT.NET.USER.ZS":        ("Utilisateurs internet",    "%",          "Demographie", True),
}


def ind_label(code):
    return IND[code][0] if code in IND else code


def ind_unit(code):
    return IND[code][1] if code in IND else ""


def ind_domain(code):
    return IND[code][2] if code in IND else ""


def ind_higher_better(code):
    return IND[code][3] if code in IND else True


# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
def inject_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    /* Header */
    .header-bar {
        background: linear-gradient(135deg, #1B3A4B 0%, #2E6F8E 60%, #1B6B50 100%);
        padding: 22px 32px;
        border-radius: 8px;
        margin-bottom: 24px;
        border-bottom: 3px solid #C9A227;
    }
    .header-bar h1 {
        color: #FFFFFF;
        margin: 0;
        font-size: 1.45em;
        font-weight: 600;
        letter-spacing: -0.3px;
    }
    .header-bar .subtitle {
        color: #B0C4CE;
        margin: 4px 0 0 0;
        font-size: 0.85em;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* KPI cards */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
        gap: 10px;
        margin-bottom: 20px;
    }
    .kpi {
        background: #161B22;
        border-radius: 6px;
        padding: 14px 16px;
        border-left: 3px solid #2E6F8E;
        transition: border-color 0.2s;
    }
    .kpi.warn { border-left-color: #B03A2E; }
    .kpi .kpi-title {
        color: #8899A6;
        font-size: 0.72em;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500;
        margin-bottom: 5px;
    }
    .kpi .kpi-val {
        color: #ECF0F1;
        font-size: 1.45em;
        font-weight: 700;
        line-height: 1.15;
    }
    .kpi .kpi-delta {
        font-size: 0.78em;
        margin-top: 3px;
        font-weight: 500;
    }
    .kpi .kpi-delta.positive { color: #3D7A5F; }
    .kpi .kpi-delta.negative { color: #B03A2E; }
    .kpi .kpi-context {
        color: #5D6D7E;
        font-size: 0.68em;
        margin-top: 2px;
    }

    /* Section headings */
    .sec-title {
        color: #D5DBE1;
        font-size: 1.0em;
        font-weight: 600;
        letter-spacing: -0.2px;
        border-bottom: 1px solid #2E6F8E;
        padding-bottom: 6px;
        margin: 22px 0 14px 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #5D6D7E;
        font-size: 0.72em;
        padding: 16px 0;
        margin-top: 28px;
        border-top: 1px solid #1E2A35;
        letter-spacing: 0.3px;
    }

    /* Sidebar refinements */
    section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stSlider label {
        font-size: 0.82em;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85em;
        font-weight: 500;
        padding: 8px 20px;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    out = {}
    for key, path in [
        ("processed", PROCESSED_PATH),
        ("predictions", PREDICTIONS_PATH),
        ("projections", PROJECTIONS_PATH),
        ("results", RESULTS_PATH),
    ]:
        if os.path.exists(path):
            out[key] = pd.read_csv(path)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt(val, unit=""):
    if pd.isna(val):
        return "—"
    if abs(val) >= 1_000_000:
        s = f"{val / 1_000_000:,.1f} M"
    elif abs(val) >= 10_000:
        s = f"{val:,.0f}"
    elif abs(val) >= 100:
        s = f"{val:,.1f}"
    else:
        s = f"{val:,.2f}"
    return f"{s} {unit}".strip() if unit else s


def uemoa_avg(df, col, year):
    sub = df[(df["year"] == year) & (df[col].notna())]
    return sub[col].mean() if not sub.empty else None


def csv_download(df, label, filename):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv")


def chart(fig, key_name):
    """Render plotly chart without deprecation warnings."""
    st.plotly_chart(fig, key=key_name)


def country_color(name):
    return COUNTRY_PAL.get(name, PAL["slate"])


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="header-bar">
        <h1>Energie et Developpement — Togo</h1>
        <p class="subtitle">Analyse predictive  ·  Zone UEMOA  ·  Cadre BCEAO</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
def render_kpis(df, focus):
    rows = df[df["country_code"] == focus].sort_values("year")
    if rows.empty:
        return

    y_max = int(rows["year"].max())
    latest = rows[rows["year"] == y_max].iloc[0]
    prev_rows = rows[rows["year"] == y_max - 1]
    prev = prev_rows.iloc[0] if not prev_rows.empty else None

    specs = [
        ("EG.USE.ELEC.KH.PC", "Consommation electrique", "kWh/hab"),
        ("EG.ELC.ACCS.ZS",    "Acces electricite",       "%"),
        ("NY.GDP.PCAP.CD",    "PIB / habitant",           "USD"),
        ("FP.CPI.TOTL.ZG",   "Inflation",                "%"),
        ("SH.DYN.MORT",      "Mortalite infantile",       "pour mille"),
        ("SP.DYN.LE00.IN",   "Esperance de vie",          "ans"),
        ("SP.POP.TOTL",      "Population",                ""),
        ("EG.FEC.RNEW.ZS",   "Renouvelables",            "%"),
    ]

    # Alert thresholds
    alerts = {
        "FP.CPI.TOTL.ZG": lambda v: v > 3,
        "SH.DYN.MORT": lambda v: v > 50,
        "EG.ELC.ACCS.ZS": lambda v: v < 50,
    }

    html = '<div class="kpi-row">'
    for col, title, unit in specs:
        if col not in latest.index or pd.isna(latest.get(col)):
            continue
        val = latest[col]
        warn = col in alerts and alerts[col](val)
        cls = "kpi warn" if warn else "kpi"

        # Delta vs N-1
        delta_html = ""
        if prev is not None and col in prev.index and pd.notna(prev.get(col)):
            d = val - prev[col]
            pct = (d / abs(prev[col])) * 100 if prev[col] != 0 else 0
            if abs(pct) > 0.05:
                hb = ind_higher_better(col)
                good = (d >= 0) == hb
                arrow = "+" if d > 0 else ""
                css = "positive" if good else "negative"
                delta_html = f'<div class="kpi-delta {css}">{arrow}{pct:.1f}% vs {y_max - 1}</div>'

        # Vs UEMOA
        avg = uemoa_avg(df, col, y_max)
        ctx_html = ""
        if avg is not None and avg != 0:
            diff = ((val - avg) / abs(avg)) * 100
            sign = "+" if diff > 0 else ""
            ctx_html = f'<div class="kpi-context">{sign}{diff:.0f}% vs moy. UEMOA</div>'

        html += f'''
        <div class="{cls}">
            <div class="kpi-title">{title}</div>
            <div class="kpi-val">{fmt(val, unit)}</div>
            {delta_html}{ctx_html}
        </div>'''

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SYNTHESE
# ─────────────────────────────────────────────────────────────────────────────
def tab_synthese(df, predictions, projections, results, focus):
    togo = df[df["country_code"] == focus].sort_values("year")
    if togo.empty:
        return
    y_max, y_min = int(togo["year"].max()), int(togo["year"].min())
    latest = togo[togo["year"] == y_max].iloc[0]
    earliest = togo[togo["year"] == y_min].iloc[0]

    st.markdown('<div class="sec-title">Synthese executive</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"##### Progression {y_min} — {y_max}")
        pairs = [
            ("EG.ELC.ACCS.ZS", "Acces electricite", "%", "+{:.1f} pts"),
            ("SH.DYN.MORT", "Mortalite infantile", "p.1000", "{:.1f}"),
            ("NY.GDP.PCAP.CD", "PIB / hab", "USD", "x{:.1f}"),
        ]
        for code, name, u, tmpl in pairs:
            if code in latest.index and code in earliest.index:
                v0, v1 = earliest.get(code), latest.get(code)
                if pd.notna(v0) and pd.notna(v1):
                    if "x" in tmpl:
                        delta_str = tmpl.format(v1 / v0) if v0 != 0 else ""
                    else:
                        delta_str = tmpl.format(v1 - v0)
                    st.markdown(f"- **{name}** : {fmt(v0, u)} → **{fmt(v1, u)}** ({delta_str})")

    with c2:
        st.markdown("##### Projections IA — 2028")
        if projections is not None and not projections.empty:
            proj_2028 = projections[
                (projections["country_code"] == focus) & (projections["year"] == 2028)
            ]
            for _, r in proj_2028.iterrows():
                st.markdown(
                    f"- **{r['target_name']}** : {r['predicted']:.1f} "
                    f"[{r['ci_lower']:.1f} – {r['ci_upper']:.1f}] {r['unit']}"
                )

    st.divider()

    # Sparklines
    st.markdown('<div class="sec-title">Tendances</div>', unsafe_allow_html=True)
    sparks = [
        ("EG.USE.ELEC.KH.PC", "Consommation electrique"),
        ("EG.ELC.ACCS.ZS", "Acces electricite"),
        ("SH.DYN.MORT", "Mortalite infantile"),
        ("NY.GDP.PCAP.CD", "PIB / habitant"),
    ]
    cols = st.columns(len(sparks))
    for i, (code, title) in enumerate(sparks):
        with cols[i]:
            if code in togo.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=togo["year"], y=togo[code],
                    mode="lines", fill="tozeroy",
                    line=dict(color=PAL["blue"], width=2),
                    fillcolor="rgba(46,111,142,0.12)",
                ))
                fig.update_layout(
                    height=110, margin=dict(l=4, r=4, t=22, b=4),
                    title=dict(text=title, font=dict(size=10, color=PAL["muted"])),
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    template=TMPL, showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                chart(fig, f"sp_{code}")

    st.divider()

    # Comparison table
    st.markdown(f'<div class="sec-title">Tableau comparatif UEMOA — {y_max}</div>', unsafe_allow_html=True)
    disp_cols = ["country_name", "EG.ELC.ACCS.ZS", "EG.USE.ELEC.KH.PC",
                 "NY.GDP.PCAP.CD", "SH.DYN.MORT", "SP.DYN.LE00.IN", "FP.CPI.TOTL.ZG"]
    avail = [c for c in disp_cols if c in df.columns]
    table = df[df["year"] == y_max][avail].copy()
    rename = {c: ind_label(c) for c in avail if c in IND}
    rename["country_name"] = "Pays"
    table = table.rename(columns=rename).sort_values("Pays").reset_index(drop=True)

    with st.expander("Voir les donnees", expanded=False):
        st.dataframe(table, height=300)
        csv_download(table, "Exporter CSV", f"uemoa_{y_max}.csv")

    # Model performance
    if results is not None and not results.empty:
        st.markdown('<div class="sec-title">Performance des modeles</div>', unsafe_allow_html=True)
        best = results.loc[results.groupby("target")["r2"].idxmax()]
        mcols = st.columns(len(best))
        for i, (_, r) in enumerate(best.iterrows()):
            color = PAL["green"] if r["r2"] > 0.85 else PAL["gold"] if r["r2"] > 0.7 else PAL["red"]
            with mcols[i]:
                st.markdown(f"""
                <div class="kpi" style="border-left-color: {color}; text-align: center;">
                    <div class="kpi-title">{r['target_name']}</div>
                    <div class="kpi-val" style="color: {color};">R² {r['r2']:.3f}</div>
                    <div class="kpi-context">{r['model']}  ·  MAE {r['mae']:.2f}  ·  RMSE {r['rmse']:.2f}</div>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ENERGIE
# ─────────────────────────────────────────────────────────────────────────────
def tab_energie(df, focus):
    st.markdown('<div class="sec-title">Analyse energetique</div>', unsafe_allow_html=True)

    togo = df[df["country_code"] == focus].sort_values("year")
    if togo.empty:
        st.warning("Aucune donnee pour ce pays.")
        return

    # Drill-down selector
    energy_codes = [c for c in IND if ind_domain(c) == "Energie" and c in df.columns]
    selected = st.selectbox(
        "Indicateur",
        energy_codes,
        format_func=lambda c: f"{ind_label(c)} ({ind_unit(c)})",
        key="nrj_sel",
    )

    left, right = st.columns([3, 1])
    with left:
        fig = go.Figure()
        for cn in df["country_name"].unique():
            sub = df[df["country_name"] == cn].sort_values("year")
            if selected not in sub.columns:
                continue
            is_f = sub["country_code"].iloc[0] == focus
            fig.add_trace(go.Scatter(
                x=sub["year"], y=sub[selected],
                name=cn, mode="lines+markers" if is_f else "lines",
                line=dict(color=country_color(cn), width=3.5 if is_f else 1.2),
                opacity=1.0 if is_f else 0.35,
            ))
        fig.update_layout(
            title=f"{ind_label(selected)} — comparaison regionale",
            xaxis_title="", yaxis_title=ind_unit(selected),
            template=TMPL, height=400, hovermode="x unified",
            legend=dict(orientation="h", y=-0.18, font_size=10),
            margin=dict(t=40),
        )
        chart(fig, "nrj_main")

    with right:
        if selected in togo.columns:
            vals = togo[selected].dropna()
            if len(vals) > 0:
                st.metric("Derniere valeur", fmt(vals.iloc[-1], ind_unit(selected)))
                if len(vals) > 1:
                    chg = vals.iloc[-1] - vals.iloc[0]
                    st.metric("Evolution totale", f"{chg:+.1f}")
            rank_col = f"{selected}_rank"
            if rank_col in togo.columns:
                y_max = int(togo["year"].max())
                rv = togo[togo["year"] == y_max][rank_col].values
                n = df[df["year"] == y_max]["country_code"].nunique()
                if len(rv) > 0:
                    st.metric("Rang UEMOA", f"{int(rv[0])} / {n}")

    st.divider()

    # Panorama 4 quadrants
    st.markdown("##### Panorama")
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Consommation (kWh/hab)", "Acces electricite (%)",
                        "Ecart urbain / rural (pts)", "Renouvelables (%)"],
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )
    quads = [
        ("EG.USE.ELEC.KH.PC", 1, 1, PAL["navy"]),
        ("EG.ELC.ACCS.ZS",    1, 2, PAL["green"]),
        ("gap_elec_urbain_rural", 2, 1, PAL["gold"]),
        ("EG.FEC.RNEW.ZS",    2, 2, PAL["teal"]),
    ]
    for code, r, c, col in quads:
        if code in togo.columns:
            fig.add_trace(go.Scatter(
                x=togo["year"], y=togo[code], mode="lines+markers",
                line=dict(color=col, width=2.2), marker=dict(size=3),
                showlegend=False,
            ), row=r, col=c)
    fig.update_layout(height=480, template=TMPL, margin=dict(t=30))
    chart(fig, "nrj_pano")

    # Urban / rural gap
    if "EG.ELC.ACCS.UR.ZS" in togo.columns and "EG.ELC.ACCS.RU.ZS" in togo.columns:
        st.markdown("##### Fracture urbain / rural")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=togo["year"], y=togo["EG.ELC.ACCS.UR.ZS"], name="Urbain",
            line=dict(color=PAL["green"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=togo["year"], y=togo["EG.ELC.ACCS.RU.ZS"], name="Rural",
            fill="tonexty", fillcolor="rgba(176,58,46,0.08)",
            line=dict(color=PAL["red"], width=2),
        ))
        fig.update_layout(
            yaxis_title="%", template=TMPL, height=320,
            hovermode="x unified", margin=dict(t=20),
            legend=dict(orientation="h", y=-0.15, font_size=10),
        )
        chart(fig, "nrj_gap")

    # Data table
    ecols = ["year"] + [c for c in energy_codes if c in togo.columns]
    t = togo[ecols].copy()
    rename = {c: ind_label(c) for c in ecols if c in IND}
    rename["year"] = "Annee"
    t = t.rename(columns=rename)
    with st.expander("Donnees detaillees", expanded=False):
        st.dataframe(t, height=300)
        csv_download(t, "Exporter CSV", "energie_togo.csv")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — IMPACT SOCIAL
# ─────────────────────────────────────────────────────────────────────────────
def tab_social(df, focus):
    st.markdown('<div class="sec-title">Impact social — energie et developpement</div>', unsafe_allow_html=True)

    togo = df[df["country_code"] == focus].sort_values("year")

    # Impact metrics
    if "EG.ELC.ACCS.ZS" in togo.columns and "SH.DYN.MORT" in togo.columns:
        pair = togo[["EG.ELC.ACCS.ZS", "SH.DYN.MORT"]].dropna()
        if len(pair) > 2:
            corr = pair["EG.ELC.ACCS.ZS"].corr(pair["SH.DYN.MORT"])
            d_acc = togo["EG.ELC.ACCS.ZS"].max() - togo["EG.ELC.ACCS.ZS"].min()
            d_mort = togo["SH.DYN.MORT"].max() - togo["SH.DYN.MORT"].min()
            impact = abs(d_mort / d_acc) if d_acc > 0 else 0

            mc = st.columns(3)
            cards = [
                ("Correlation acces / mortalite", f"{corr:.3f}", "Pearson", PAL["blue"]),
                ("Impact estime", f"{impact:.2f}", "pts mortalite / pt acces electr.", PAL["green"]),
                ("Progres acces", f"+{d_acc:.1f} pts", "Electrification sur la periode", PAL["teal"]),
            ]
            for i, (title, val, ctx, color) in enumerate(cards):
                with mc[i]:
                    st.markdown(f"""
                    <div class="kpi" style="border-left-color: {color};">
                        <div class="kpi-title">{title}</div>
                        <div class="kpi-val">{val}</div>
                        <div class="kpi-context">{ctx}</div>
                    </div>""", unsafe_allow_html=True)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        # Scatter UEMOA
        if "EG.ELC.ACCS.ZS" in df.columns and "SH.DYN.MORT" in df.columns:
            y_max = int(df["year"].max())
            snap = df[df["year"] == y_max].dropna(subset=["EG.ELC.ACCS.ZS", "SH.DYN.MORT"])
            fig = px.scatter(
                snap, x="EG.ELC.ACCS.ZS", y="SH.DYN.MORT",
                color="country_name",
                size="SP.POP.TOTL" if "SP.POP.TOTL" in snap.columns else None,
                hover_name="country_name",
                color_discrete_map=COUNTRY_PAL, template=TMPL,
                labels={"EG.ELC.ACCS.ZS": "Acces electricite (%)",
                        "SH.DYN.MORT": "Mortalite infantile (p. 1000)"},
            )
            if len(snap) > 2:
                valid = snap[["EG.ELC.ACCS.ZS", "SH.DYN.MORT"]].dropna()
                z = np.polyfit(valid["EG.ELC.ACCS.ZS"], valid["SH.DYN.MORT"], 1)
                xr = np.linspace(valid["EG.ELC.ACCS.ZS"].min(), valid["EG.ELC.ACCS.ZS"].max(), 80)
                fig.add_trace(go.Scatter(
                    x=xr, y=z[0] * xr + z[1], mode="lines", name="Tendance",
                    line=dict(dash="dash", color=PAL["muted"], width=1.2),
                ))
            fig.update_layout(
                title=f"Acces electricite vs mortalite — {y_max}",
                height=400, margin=dict(t=40),
                legend=dict(font_size=9),
            )
            chart(fig, "soc_scatter")

    with c2:
        if not togo.empty and "EG.ELC.ACCS.ZS" in togo.columns and "SH.DYN.MORT" in togo.columns:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=togo["year"], y=togo["EG.ELC.ACCS.ZS"], name="Acces electr. (%)",
                line=dict(color=PAL["green"], width=2.5),
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=togo["year"], y=togo["SH.DYN.MORT"], name="Mortalite (p. 1000)",
                line=dict(color=PAL["red"], width=2.5),
            ), secondary_y=True)
            fig.update_layout(
                title="Electricite et mortalite — evolution croisee",
                template=TMPL, height=400, hovermode="x unified", margin=dict(t=40),
                legend=dict(orientation="h", y=-0.15, font_size=10),
            )
            fig.update_yaxes(title_text="%", secondary_y=False)
            fig.update_yaxes(title_text="p. 1000", secondary_y=True)
            chart(fig, "soc_dual")

    # Water & sanitation + life expectancy
    st.markdown("##### Indicateurs complementaires")
    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        for code, name, color in [
            ("SH.H2O.BASW.ZS", "Eau potable", PAL["blue"]),
            ("SH.STA.BASS.ZS", "Assainissement", PAL["orange"]),
        ]:
            if code in togo.columns:
                fig.add_trace(go.Scatter(
                    x=togo["year"], y=togo[code], name=name,
                    mode="lines+markers", line=dict(color=color, width=2),
                ))
        fig.update_layout(
            title="Eau potable et assainissement",
            template=TMPL, height=340, hovermode="x unified", margin=dict(t=40),
            legend=dict(orientation="h", y=-0.15, font_size=10),
        )
        chart(fig, "soc_water")

    with c4:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if "SP.DYN.LE00.IN" in togo.columns:
            fig.add_trace(go.Scatter(
                x=togo["year"], y=togo["SP.DYN.LE00.IN"], name="Esperance de vie (ans)",
                line=dict(color=PAL["green"], width=2),
            ), secondary_y=False)
        if "SH.XPD.CHEX.PC.CD" in togo.columns:
            fig.add_trace(go.Bar(
                x=togo["year"], y=togo["SH.XPD.CHEX.PC.CD"], name="Depenses sante ($/hab)",
                marker_color=PAL["blue"], opacity=0.4,
            ), secondary_y=True)
        fig.update_layout(
            title="Esperance de vie et depenses de sante",
            template=TMPL, height=340, margin=dict(t=40),
            legend=dict(orientation="h", y=-0.15, font_size=10),
        )
        chart(fig, "soc_health")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ECONOMIE
# ─────────────────────────────────────────────────────────────────────────────
def tab_economie(df, focus):
    st.markdown('<div class="sec-title">Indicateurs macroeconomiques — surveillance BCEAO</div>', unsafe_allow_html=True)

    togo = df[df["country_code"] == focus].sort_values("year")
    if togo.empty:
        return

    econ_codes = [c for c in IND if ind_domain(c) == "Economie" and c in df.columns]
    selected = st.selectbox(
        "Indicateur",
        econ_codes,
        format_func=lambda c: f"{ind_label(c)} ({ind_unit(c)})",
        key="eco_sel",
    )

    fig = go.Figure()
    for cn in df["country_name"].unique():
        sub = df[df["country_name"] == cn].sort_values("year")
        if selected not in sub.columns:
            continue
        is_f = sub["country_code"].iloc[0] == focus
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub[selected],
            name=cn, mode="lines+markers" if is_f else "lines",
            line=dict(color=country_color(cn), width=3.5 if is_f else 1.2),
            opacity=1.0 if is_f else 0.35,
        ))
    if selected == "FP.CPI.TOTL.ZG":
        fig.add_hline(y=3, line_dash="dash", line_color=PAL["red"],
                      annotation_text="Cible BCEAO (3%)",
                      annotation_font_color=PAL["red"], annotation_font_size=10)
    fig.update_layout(
        title=f"{ind_label(selected)} — comparaison regionale",
        yaxis_title=ind_unit(selected), template=TMPL, height=400,
        hovermode="x unified", margin=dict(t=40),
        legend=dict(orientation="h", y=-0.18, font_size=10),
    )
    chart(fig, "eco_drill")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if "NY.GDP.PCAP.CD" in togo.columns:
            fig.add_trace(go.Scatter(
                x=togo["year"], y=togo["NY.GDP.PCAP.CD"], name="PIB/hab (USD)",
                mode="lines+markers", line=dict(color=PAL["navy"], width=2.2),
            ), secondary_y=False)
        if "NY.GDP.MKTP.KD.ZG" in togo.columns:
            colors = [PAL["green"] if v >= 0 else PAL["red"] for v in togo["NY.GDP.MKTP.KD.ZG"]]
            fig.add_trace(go.Bar(
                x=togo["year"], y=togo["NY.GDP.MKTP.KD.ZG"], name="Croissance (%)",
                marker_color=colors, opacity=0.55,
            ), secondary_y=True)
        fig.update_layout(title="PIB et croissance", template=TMPL, height=370, margin=dict(t=40),
                          legend=dict(orientation="h", y=-0.15, font_size=10))
        fig.update_yaxes(title_text="USD", secondary_y=False)
        fig.update_yaxes(title_text="%", secondary_y=True)
        chart(fig, "eco_gdp")

    with c2:
        if "NE.EXP.GNFS.ZS" in togo.columns and "NE.IMP.GNFS.ZS" in togo.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=togo["year"], y=togo["NE.EXP.GNFS.ZS"], name="Exports (% PIB)",
                line=dict(color=PAL["green"], width=2),
            ))
            fig.add_trace(go.Scatter(
                x=togo["year"], y=togo["NE.IMP.GNFS.ZS"], name="Imports (% PIB)",
                line=dict(color=PAL["red"], width=2),
            ))
            if "balance_commerciale" in togo.columns:
                clrs = [PAL["green"] if v >= 0 else PAL["red"] for v in togo["balance_commerciale"]]
                fig.add_trace(go.Bar(
                    x=togo["year"], y=togo["balance_commerciale"], name="Balance",
                    marker_color=clrs, opacity=0.25,
                ))
            fig.update_layout(title="Commerce exterieur (% PIB)", template=TMPL, height=370,
                              hovermode="x unified", margin=dict(t=40),
                              legend=dict(orientation="h", y=-0.15, font_size=10))
            chart(fig, "eco_trade")

    with st.expander("Donnees detaillees", expanded=False):
        ecols_t = ["year"] + [c for c in econ_codes if c in togo.columns]
        t = togo[ecols_t].copy()
        rename = {c: ind_label(c) for c in ecols_t if c in IND}
        rename["year"] = "Annee"
        t = t.rename(columns=rename)
        st.dataframe(t, height=300)
        csv_download(t, "Exporter CSV", "economie_togo.csv")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
def tab_predictions(predictions, projections, results, focus):
    st.markdown('<div class="sec-title">Predictions et projections</div>', unsafe_allow_html=True)

    if predictions is None or predictions.empty:
        st.info("Executez les modeles : python src/models/train.py && python src/models/predict.py")
        return

    targets = predictions["target"].unique()
    target_map = {t: predictions[predictions["target"] == t]["target_name"].iloc[0] for t in targets}
    sel = st.selectbox("Cible", targets, format_func=lambda t: target_map[t], key="pred_sel")

    tp = predictions[predictions["target"] == sel]
    togo_p = tp[tp["country_code"] == focus]
    unit = tp["unit"].iloc[0]

    fig = go.Figure()
    if not togo_p.empty:
        fig.add_trace(go.Scatter(
            x=togo_p["year"], y=togo_p["actual"], name="Observe",
            mode="lines+markers", line=dict(color=PAL["navy"], width=2.5),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=togo_p["year"], y=togo_p["predicted"], name="Predit (ML)",
            mode="lines+markers", line=dict(color=PAL["gold"], width=2, dash="dash"),
            marker=dict(size=4),
        ))

    if projections is not None:
        tp_proj = projections[
            (projections["target"] == sel) & (projections["country_code"] == focus)
        ]
        if not tp_proj.empty:
            fig.add_trace(go.Scatter(
                x=tp_proj["year"], y=tp_proj["predicted"], name="Projection",
                mode="lines+markers", line=dict(color=PAL["teal"], width=2.5),
                marker=dict(size=6, symbol="diamond"),
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([tp_proj["year"], tp_proj["year"][::-1]]),
                y=pd.concat([tp_proj["ci_upper"], tp_proj["ci_lower"][::-1]]),
                fill="toself", fillcolor="rgba(58,158,146,0.10)",
                line=dict(color="rgba(0,0,0,0)"), name="IC 95%",
            ))

    fig.update_layout(
        title=f"{target_map[sel]} — historique et projections",
        xaxis_title="", yaxis_title=f"{target_map[sel]} ({unit})",
        template=TMPL, height=430, hovermode="x unified", margin=dict(t=40),
        legend=dict(orientation="h", y=-0.15, font_size=10),
    )
    chart(fig, "pred_main")

    # Performance metrics
    if not togo_p.empty:
        mc = st.columns(4)
        mae = togo_p["error"].abs().mean()
        rmse = np.sqrt((togo_p["error"] ** 2).mean())
        mape_val = togo_p["error_pct"].abs().mean() if "error_pct" in togo_p.columns else np.nan
        best_r2 = results[results["target"] == sel]["r2"].max() if results is not None else np.nan
        with mc[0]:
            st.metric("MAE", f"{mae:.2f} {unit}")
        with mc[1]:
            st.metric("RMSE", f"{rmse:.2f}")
        with mc[2]:
            if not pd.isna(mape_val):
                st.metric("MAPE", f"{mape_val:.1f}%")
        with mc[3]:
            if not pd.isna(best_r2):
                st.metric("Meilleur R²", f"{best_r2:.4f}")

    # Projections table
    if projections is not None:
        proj_disp = projections[
            (projections["target"] == sel) & (projections["country_code"] == focus)
        ][["year", "predicted", "ci_lower", "ci_upper"]].copy()
        proj_disp.columns = ["Annee", f"Predit ({unit})", f"IC bas ({unit})", f"IC haut ({unit})"]
        with st.expander("Projections detaillees", expanded=False):
            st.dataframe(proj_disp, height=220)

    # Model comparison
    if results is not None and not results.empty:
        st.markdown("##### Comparaison des algorithmes")
        tr = results[results["target"] == sel]
        fig = go.Figure()
        bar_colors = [PAL["green"] if r > 0.85 else PAL["gold"] if r > 0.7 else PAL["red"] for r in tr["r2"]]
        fig.add_trace(go.Bar(
            x=tr["model"], y=tr["r2"], marker_color=bar_colors,
            text=[f"{r:.3f}" for r in tr["r2"]], textposition="outside",
            textfont_size=11,
        ))
        fig.update_layout(
            title=f"R² par modele — {target_map[sel]}",
            yaxis_title="R²", yaxis_range=[0, 1.08],
            template=TMPL, height=340, margin=dict(t=40),
        )
        chart(fig, "pred_cmp")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — BENCHMARK UEMOA
# ─────────────────────────────────────────────────────────────────────────────
def tab_benchmark(df, focus):
    st.markdown('<div class="sec-title">Positionnement regional — UEMOA</div>', unsafe_allow_html=True)

    y_max = int(df["year"].max())
    snap = df[df["year"] == y_max].copy()

    # Radar
    radar_specs = [
        ("EG.ELC.ACCS.ZS", "Acces electr."),
        ("NY.GDP.PCAP.CD", "PIB/hab"),
        ("SP.DYN.LE00.IN", "Esp. vie"),
        ("SH.H2O.BASW.ZS", "Eau potable"),
        ("IT.NET.USER.ZS", "Internet"),
    ]
    avail_r = [(c, l) for c, l in radar_specs if c in snap.columns]

    if len(avail_r) >= 3:
        st.markdown("##### Profil multidimensionnel")
        fig = go.Figure()
        for _, row in snap.iterrows():
            cn = row["country_name"]
            vals = []
            for code, _ in avail_r:
                v = row.get(code, np.nan)
                mx = snap[code].max()
                vals.append((v / mx * 100) if pd.notna(v) and mx > 0 else 0)
            vals.append(vals[0])
            labels = [l for _, l in avail_r] + [avail_r[0][1]]
            is_f = row["country_code"] == focus
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=labels, name=cn,
                line=dict(color=country_color(cn), width=3 if is_f else 0.8),
                fill="toself" if is_f else None,
                fillcolor="rgba(27,107,80,0.12)" if is_f else None,
                opacity=1.0 if is_f else 0.4,
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 108], showticklabels=False, gridcolor="#1E2A35"),
                angularaxis=dict(gridcolor="#1E2A35"),
                bgcolor="rgba(0,0,0,0)",
            ),
            template=TMPL, height=480, showlegend=True,
            title=f"Profil UEMOA — {y_max} (normalise 0-100)",
            legend=dict(orientation="h", y=-0.12, font_size=9),
            margin=dict(t=50),
        )
        chart(fig, "bench_radar")

    st.divider()

    # Classements
    st.markdown("##### Classements")
    rank_specs = [
        ("EG.ELC.ACCS.ZS", "Acces electricite (%)", True),
        ("EG.USE.ELEC.KH.PC", "Consommation (kWh/hab)", True),
        ("SH.DYN.MORT", "Mortalite infantile (p. 1000)", False),
        ("NY.GDP.PCAP.CD", "PIB / habitant (USD)", True),
    ]
    cols = st.columns(2)
    for i, (code, title, asc) in enumerate(rank_specs):
        if code not in snap.columns:
            continue
        with cols[i % 2]:
            s = snap.dropna(subset=[code]).sort_values(code, ascending=not asc)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=s[code], y=s["country_name"], orientation="h",
                marker_color=[country_color(n) for n in s["country_name"]],
                text=[f"{v:.1f}" for v in s[code]], textposition="outside",
                textfont_size=10,
            ))
            fig.update_layout(
                title=title, template=TMPL, height=280,
                margin=dict(l=10, r=50, t=35, b=10),
                yaxis=dict(autorange="reversed"),
                xaxis=dict(showticklabels=False),
            )
            chart(fig, f"bench_{code}")

    st.divider()

    # Evolution comparee
    st.markdown("##### Evolution temporelle")
    compare_codes = [c for c in IND if c in df.columns]
    compare_sel = st.selectbox(
        "Indicateur", compare_codes,
        format_func=lambda c: f"{ind_label(c)} ({ind_unit(c)})",
        key="bench_evo",
    )
    fig = px.line(
        df.dropna(subset=[compare_sel]), x="year", y=compare_sel,
        color="country_name", color_discrete_map=COUNTRY_PAL,
        labels={"year": "Annee", compare_sel: ind_label(compare_sel), "country_name": "Pays"},
        template=TMPL,
    )
    for trace in fig.data:
        if trace.name == df[df["country_code"] == focus]["country_name"].iloc[0]:
            trace.line.width = 3.5
        else:
            trace.line.width = 1.2
            trace.opacity = 0.4
    fig.update_layout(
        height=400, hovermode="x unified", margin=dict(t=20),
        legend=dict(orientation="h", y=-0.15, font_size=10),
    )
    chart(fig, "bench_evo_chart")

    # Table
    rank_cols = ["country_name"] + [c for c, _, _ in rank_specs if c in snap.columns]
    t = snap[rank_cols].copy()
    rename = {c: ind_label(c) for c in rank_cols if c in IND}
    rename["country_name"] = "Pays"
    t = t.rename(columns=rename).sort_values("Pays").reset_index(drop=True)
    with st.expander("Donnees comparatives", expanded=False):
        st.dataframe(t, height=300)
        csv_download(t, "Exporter CSV", f"benchmark_uemoa_{y_max}.csv")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_style()
    render_header()

    data = load_data()
    if "processed" not in data:
        st.error(
            "Donnees introuvables. Executez le pipeline :\n\n"
            "python src/etl/extract.py\n"
            "python src/etl/transform.py\n"
            "python src/models/train.py\n"
            "python src/models/predict.py"
        )
        return

    df = data["processed"]
    predictions = data.get("predictions")
    projections = data.get("projections")
    results = data.get("results")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Filtres")
        st.divider()

        all_codes = sorted(df["country_code"].unique())
        focus = st.selectbox(
            "Pays principal",
            all_codes,
            format_func=lambda c: df[df["country_code"] == c]["country_name"].iloc[0],
            index=all_codes.index("TG") if "TG" in all_codes else 0,
        )

        all_names = sorted(df["country_name"].unique())
        sel_countries = st.multiselect("Pays affiches", all_names, default=all_names)

        y_min, y_max = int(df["year"].min()), int(df["year"].max())
        yr = st.slider("Periode", y_min, y_max, (y_min, y_max))

        domain = st.selectbox("Domaine", ["Tous", "Energie", "Economie", "Sante", "Demographie"])

        st.divider()

        st.markdown("##### Exports")
        csv_download(df, "Donnees completes", "uemoa_complet.csv")
        if predictions is not None:
            csv_download(predictions, "Predictions", "predictions.csv")
        if projections is not None:
            csv_download(projections, "Projections 2024-2028", "projections.csv")

        st.divider()
        focus_name = df[df["country_code"] == focus]["country_name"].iloc[0]
        st.markdown(
            f"**Source** : Banque Mondiale (WDI)\n\n"
            f"**Focus** : {focus_name}\n\n"
            f"**Periode** : {yr[0]} – {yr[1]}\n\n"
            f"**Observations** : {len(df):,}\n\n"
            f"**Indicateurs** : {len([c for c in df.columns if c in IND])}"
        )

    # Apply filters
    filtered = df[
        (df["country_name"].isin(sel_countries)) & (df["year"].between(*yr))
    ]
    if domain != "Tous":
        keep = [c for c in IND if ind_domain(c) == domain and c in filtered.columns]
        meta = ["country_code", "country_name", "year"]
        extra = [c for c in filtered.columns if c not in IND and c not in meta]
        filtered = filtered[[c for c in meta + keep + extra if c in filtered.columns]]

    # KPIs
    render_kpis(df, focus)

    # Tabs
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Synthese",
        "Energie",
        "Impact social",
        "Economie",
        "Predictions",
        "Benchmark UEMOA",
    ])

    with t1:
        tab_synthese(filtered, predictions, projections, results, focus)
    with t2:
        tab_energie(filtered, focus)
    with t3:
        tab_social(filtered, focus)
    with t4:
        tab_economie(filtered, focus)
    with t5:
        tab_predictions(predictions, projections, results, focus)
    with t6:
        tab_benchmark(filtered, focus)

    # Footer
    st.markdown(
        '<div class="footer">'
        "Source : Banque Mondiale Open Data (WDI)  |  "
        "Modeles : scikit-learn, XGBoost, LightGBM  |  "
        "Cadre : BCEAO / UEMOA  |  "
        "Dashboard : Streamlit + Plotly"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
