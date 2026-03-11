"""
Dashboard IA — Anticipation de la Demande Electrique au Togo
Developpeur en Intelligence Artificielle | BCEAO
Pipeline : Extraction (Banque Mondiale) -> Transformation -> Modele IA -> Predictions 2045
"""
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IA — Demande Electrique Togo 2045",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMPL = "plotly_dark"
FOCUS = "TG"

C = {
    "pop": "#2E86C1", "kwh": "#E67E22", "gwh": "#1ABC9C", "proj": "#9B59B6",
    "ci": "rgba(155,89,182,0.15)", "grid": "#1E2A35", "muted": "#7F8C8D",
    "good": "#27AE60", "warn": "#E74C3C", "bg": "#0E1117", "accent": "#3498DB",
    "ind": "#F39C12", "soc": "#E91E63", "eco": "#00BCD4",
}

IND_LABELS = {
    "SP.POP.TOTL": "Population totale", "SP.POP.GROW": "Croissance demo. (%)",
    "SP.URB.TOTL.IN.ZS": "Urbanisation (%)", "SP.DYN.TFRT.IN": "Fecondite",
    "SP.DYN.LE00.IN": "Esperance de vie", "SP.POP.0014.TO.ZS": "Pop 0-14 (%)",
    "SP.POP.1564.TO.ZS": "Pop 15-64 (%)", "EG.USE.ELEC.KH.PC": "kWh/hab",
    "EG.ELC.ACCS.ZS": "Acces electr. (%)", "EG.ELC.ACCS.UR.ZS": "Acces urbain (%)",
    "EG.ELC.ACCS.RU.ZS": "Acces rural (%)", "EG.FEC.RNEW.ZS": "Renouvelable (%)",
    "EG.USE.PCAP.KG.OE": "Energie (kg petrole/hab)", "NY.GDP.PCAP.CD": "PIB/hab (USD)",
    "NY.GDP.MKTP.CD": "PIB total (USD)", "NY.GDP.MKTP.KD.ZG": "Croissance PIB (%)",
    "NV.IND.TOTL.ZS": "Industrie (% PIB)", "FP.CPI.TOTL.ZG": "Inflation (%)",
    "IT.CEL.SETS.P2": "Mobile (/100 hab)", "SE.ADT.LITR.ZS": "Alphabetisation (%)",
    "SL.UEM.TOTL.ZS": "Chomage (%)",
}

IND_CAT = {
    "SP.POP.TOTL": "demo", "SP.POP.GROW": "demo", "SP.URB.TOTL.IN.ZS": "demo",
    "SP.DYN.TFRT.IN": "demo", "SP.DYN.LE00.IN": "demo", "SP.POP.0014.TO.ZS": "demo",
    "SP.POP.1564.TO.ZS": "demo", "EG.USE.ELEC.KH.PC": "energie",
    "EG.ELC.ACCS.ZS": "energie", "EG.ELC.ACCS.UR.ZS": "energie",
    "EG.ELC.ACCS.RU.ZS": "energie", "EG.FEC.RNEW.ZS": "energie",
    "EG.USE.PCAP.KG.OE": "energie", "NY.GDP.PCAP.CD": "eco",
    "NY.GDP.MKTP.CD": "eco", "NY.GDP.MKTP.KD.ZG": "eco",
    "NV.IND.TOTL.ZS": "eco", "FP.CPI.TOTL.ZG": "eco",
    "IT.CEL.SETS.P2": "social", "SE.ADT.LITR.ZS": "social", "SL.UEM.TOTL.ZS": "social",
}
CAT_COLORS = {"demo": C["pop"], "energie": C["kwh"], "eco": C["eco"], "social": C["soc"]}
CAT_NAMES = {"demo": "Demographie", "energie": "Energie", "eco": "Economie", "social": "Social"}


# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 0.8rem; max-width: 1400px; }

.hdr {
    background: linear-gradient(135deg, #0D2137 0%, #1B6B50 100%);
    padding: 20px 30px; border-radius: 8px; margin-bottom: 18px;
    border-bottom: 3px solid #E67E22;
}
.hdr h1 { color: #fff; margin: 0; font-size: 1.35em; font-weight: 700; }
.hdr p  { color: #B0C4CE; margin: 4px 0 0; font-size: 0.78em; font-weight: 300; }
.hdr .tag { display: inline-block; background: rgba(52,152,219,0.2); color: #3498DB;
            padding: 2px 10px; border-radius: 10px; font-size: 0.7em; font-weight: 600;
            margin-top: 6px; }

.card {
    background: #161B22; border-radius: 6px; padding: 13px 15px;
    border-left: 3px solid #2E86C1;
}
.card .t { color: #7F8C8D; font-size: 0.66em; text-transform: uppercase;
           letter-spacing: 0.8px; margin-bottom: 3px; }
.card .v { color: #ECF0F1; font-size: 1.3em; font-weight: 700; line-height: 1.15; }
.card .d { font-size: 0.74em; margin-top: 3px; font-weight: 500; }
.card .d.up { color: #27AE60; }
.card .d.dn { color: #E74C3C; }
.card .ctx { color: #5D6D7E; font-size: 0.64em; margin-top: 2px; }

.sec { color: #D5DBE1; font-size: 1.02em; font-weight: 600;
       border-bottom: 1px solid #2E6F8E;
       padding-bottom: 5px; margin: 20px 0 10px 0; }

.insight {
    background: linear-gradient(135deg, #161B22, #1a2332);
    border-left: 3px solid #3498DB; border-radius: 0 6px 6px 0;
    padding: 12px 18px; margin: 8px 0 16px 0;
    color: #B0C4CE; font-size: 0.84em; line-height: 1.55;
}
.insight strong { color: #ECF0F1; }
.insight .val { color: #1ABC9C; font-weight: 600; }
.insight .warn { color: #E74C3C; font-weight: 600; }
.insight .up { color: #27AE60; font-weight: 600; }

.step-box {
    background: #161B22; border: 1px solid #2E6F8E; border-radius: 6px;
    padding: 12px 16px; margin-bottom: 12px;
}
.step-box .step-n { color: #3498DB; font-size: 0.7em; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.step-box .step-t { color: #ECF0F1; font-weight: 600; font-size: 0.9em; }
.step-box .step-d { color: #7F8C8D; font-size: 0.74em; margin-top: 2px; }

.foot { text-align: center; color: #5D6D7E; font-size: 0.68em; padding: 14px 0;
        margin-top: 20px; border-top: 1px solid #1E2A35; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    d = {}
    for k, f in [("df", "data/processed/energy_data_processed.csv"),
                  ("raw", "data/raw/energy_data_raw.csv"),
                  ("pred", "data/predictions/predictions.csv"),
                  ("proj", "data/predictions/projections.csv"),
                  ("res", "models/results.csv")]:
        p = os.path.join(BASE, f)
        if os.path.exists(p):
            d[k] = pd.read_csv(p)
    return d


def fmt(v, u=""):
    if pd.isna(v):
        return "—"
    if abs(v) >= 1e9:
        s = f"{v/1e9:,.2f} Mrd"
    elif abs(v) >= 1e6:
        s = f"{v/1e6:,.1f} M"
    elif abs(v) >= 1e3:
        s = f"{v:,.0f}"
    else:
        s = f"{v:,.1f}"
    return f"{s} {u}".strip() if u else s


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
    <h1>Anticipation de la Demande Electrique — Togo</h1>
    <p>La population croit, combien d'electricite faudra-t-il demain ?
       &nbsp;|&nbsp; Source : Banque Mondiale (WDI) &nbsp;|&nbsp; Horizon 2045</p>
    <span class="tag">Developpeur en Intelligence Artificielle — BCEAO</span>
</div>
""", unsafe_allow_html=True)

data = load()
if "df" not in data:
    st.error("Donnees absentes. Executez le pipeline ETL d'abord.")
    st.stop()

df_all = data["df"]
df = df_all[df_all["country_code"] == FOCUS].copy()
raw_all = data.get("raw")
raw_tg = raw_all[raw_all["country_code"] == FOCUS].copy() if raw_all is not None else None
pred_all = data.get("pred")
pred = pred_all[pred_all["country_code"] == FOCUS].copy() if pred_all is not None else None
proj_all = data.get("proj")
proj = proj_all[proj_all["country_code"] == FOCUS].copy() if proj_all is not None else None
res = data.get("res")

if df.empty:
    st.error("Aucune donnee pour le Togo.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filtres")
    y_min_h, y_max_h = int(df["year"].min()), int(df["year"].max())
    yr = st.slider("Periode historique", y_min_h, y_max_h, (y_min_h, y_max_h), key="yr_hist")

    proj_years = sorted(proj["year"].unique().astype(int).tolist()) if proj is not None and not proj.empty else []
    proj_yr = st.select_slider("Horizon projection", options=proj_years,
                                value=proj_years[-1], key="yr_proj") if proj_years else None

    st.divider()
    st.markdown(f"**Historique** : {yr[0]} — {yr[1]}")
    if proj_yr:
        st.markdown(f"**Projection** : jusqu'a {proj_yr}")

    st.divider()
    st.markdown("##### Telecharger")
    if raw_all is not None:
        st.download_button("Donnees brutes", raw_all.to_csv(index=False).encode(),
                           "donnees_brutes.csv", key="dl_raw")
    if pred is not None:
        st.download_button("Predictions", pred.to_csv(index=False).encode(),
                           "predictions.csv", key="dl_pred")
    if proj is not None:
        st.download_button("Projections 2045", proj.to_csv(index=False).encode(),
                           "projections_2045.csv", key="dl_proj")

tg = df[df["year"].between(*yr)].sort_values("year")

# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
if not tg.empty:
    last, first = tg.iloc[-1], tg.iloc[0]
    cards = []

    if "SP.POP.TOTL" in tg.columns:
        pop_now, pop_bef = last["SP.POP.TOTL"], first["SP.POP.TOTL"]
        gr = ((pop_now / pop_bef) - 1) * 100 if pop_bef > 0 else 0
        cards.append(("Population", fmt(pop_now), f"+{gr:.0f}% depuis {yr[0]}", "up", C["pop"]))
    if "conso_totale_gwh" in tg.columns:
        gwh, gwh0 = last["conso_totale_gwh"], first["conso_totale_gwh"]
        d = ((gwh / gwh0) - 1) * 100 if gwh0 > 0 else 0
        cards.append(("Demande electrique", fmt(gwh, "GWh"), f"+{d:.0f}% depuis {yr[0]}", "up", C["gwh"]))
    if "EG.ELC.ACCS.ZS" in tg.columns:
        acc, acc0 = last["EG.ELC.ACCS.ZS"], first["EG.ELC.ACCS.ZS"]
        cards.append(("Acces electrique", f"{acc:.1f}%", f"{acc-acc0:+.1f} pts", "up" if acc > acc0 else "dn", C["good"]))
    if proj_yr and proj is not None and not proj.empty:
        row_p = proj[proj["year"] == proj_yr]
        if not row_p.empty:
            cards.append(("Prediction " + str(proj_yr), fmt(row_p.iloc[0]["predicted_gwh"], "GWh"),
                          "Projection IA", "up", C["proj"]))
    if res is not None and not res.empty:
        best = res.sort_values("r2", ascending=False).iloc[0]
        cards.append(("Modele IA", f"R2 = {best['r2']:.3f}", best["model"], "up", C["accent"]))

    html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:8px;margin-bottom:14px;">'
    for title, val, delta, css, color in cards:
        html += f'''<div class="card" style="border-left-color:{color};">
            <div class="t">{title}</div><div class="v">{val}</div>
            <div class="d {css}">{delta}</div></div>'''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "1. Donnees collectees",
    "2. Exploration",
    "3. Modele IA",
    "4. Predictions 2045",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DONNEES COLLECTEES  (graphiques, pas de tableaux)
# ═════════════════════════════════════════════════════════════════════════════
with t1:
    n_ind = len(raw_all["indicator_code"].unique()) if raw_all is not None else 0
    n_pays = len(raw_all["country_code"].unique()) if raw_all is not None else 0
    n_raw = len(raw_all) if raw_all is not None else 0
    yr_min_raw = int(raw_all["year"].min()) if raw_all is not None else 0
    yr_max_raw = int(raw_all["year"].max()) if raw_all is not None else 0

    st.markdown(f"""
    <div class="step-box">
        <div class="step-n">Etape 1 — Extraction</div>
        <div class="step-t">Collecte de donnees via l'API Banque Mondiale</div>
        <div class="step-d">{n_ind} indicateurs &times; {n_pays} pays UEMOA &times; {yr_max_raw - yr_min_raw + 1} annees
              = <strong>{n_raw:,} observations</strong> extraites</div>
    </div>""", unsafe_allow_html=True)

    if raw_all is not None and not raw_all.empty:
        raw_f = raw_all[raw_all["year"].between(*yr)]

        # --- Graph 1: Repartition des indicateurs par categorie ---
        st.markdown('<div class="sec">Repartition des indicateurs par domaine</div>',
                    unsafe_allow_html=True)
        cat_data = []
        for code in raw_f["indicator_code"].unique():
            cat = IND_CAT.get(code, "autre")
            cat_data.append({"Domaine": CAT_NAMES.get(cat, cat), "code": code,
                             "Indicateur": IND_LABELS.get(code, code)})
        cat_df = pd.DataFrame(cat_data)
        cat_counts = cat_df.groupby("Domaine").size().reset_index(name="Nombre")

        c1, c2 = st.columns([2, 3])
        with c1:
            fig = go.Figure(go.Pie(
                labels=cat_counts["Domaine"], values=cat_counts["Nombre"],
                hole=0.55, marker_colors=[C["pop"], C["eco"], C["kwh"], C["soc"]],
                textinfo="label+value", textfont_size=11,
            ))
            fig.update_layout(
                title="21 indicateurs collectes",
                template=TMPL, height=320, margin=dict(t=40, b=10),
                showlegend=False,
            )
            fig.add_annotation(text=f"<b>{n_ind}</b><br>indicateurs",
                               x=0.5, y=0.5, font_size=14, showarrow=False,
                               font_color="#ECF0F1")
            st.plotly_chart(fig, key="cat_pie")

        with c2:
            # Nombre d'observations par indicateur
            obs_per_ind = raw_f.groupby("indicator_code").size().reset_index(name="obs")
            obs_per_ind["label"] = obs_per_ind["indicator_code"].map(
                lambda c: IND_LABELS.get(c, c))
            obs_per_ind["color"] = obs_per_ind["indicator_code"].map(
                lambda c: CAT_COLORS.get(IND_CAT.get(c, "autre"), C["muted"]))
            obs_per_ind = obs_per_ind.sort_values("obs")

            fig = go.Figure(go.Bar(
                x=obs_per_ind["obs"], y=obs_per_ind["label"],
                orientation="h", marker_color=obs_per_ind["color"],
                text=obs_per_ind["obs"], textposition="outside", textfont_size=9,
            ))
            fig.update_layout(
                title="Volume d'observations par indicateur",
                template=TMPL, height=520, margin=dict(l=10, r=40, t=40, b=10),
                xaxis=dict(showticklabels=False),
            )
            st.plotly_chart(fig, key="obs_bar")

        st.markdown(f"""
        <div class="insight">
            <strong>Interpretation</strong> — Le pipeline extrait <span class="val">{n_raw:,} observations</span>
            couvrant {n_ind} indicateurs de la Banque Mondiale pour {n_pays} pays de la zone UEMOA
            sur la periode {yr_min_raw}-{yr_max_raw}. Les indicateurs de <strong>demographie</strong> (7) et
            d'<strong>energie</strong> (6) constituent le socle du modele. Les indicateurs
            <strong>economiques</strong> (5) et <strong>sociaux</strong> (3) apportent un contexte
            structurel qui ameliore la precision de la prediction.
        </div>""", unsafe_allow_html=True)

        st.divider()

        # --- Graph 2: Couverture temporelle par pays ---
        st.markdown('<div class="sec">Couverture des donnees par pays UEMOA</div>',
                    unsafe_allow_html=True)
        cov = raw_f.groupby("country_code").agg(
            pays=("country_name", "first"),
            observations=("value", "count"),
            indicateurs=("indicator_code", "nunique"),
        ).reset_index()

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Observations par pays",
                            "Indicateurs disponibles par pays"],
                            horizontal_spacing=0.12)
        cov_s = cov.sort_values("observations")
        colors_pays = [C["gwh"] if c == FOCUS else C["muted"] for c in cov_s["country_code"]]
        fig.add_trace(go.Bar(
            x=cov_s["observations"], y=cov_s["pays"], orientation="h",
            marker_color=colors_pays, text=cov_s["observations"],
            textposition="outside", textfont_size=10, showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=cov_s["indicateurs"], y=cov_s["pays"], orientation="h",
            marker_color=colors_pays, text=cov_s["indicateurs"],
            textposition="outside", textfont_size=10, showlegend=False,
        ), row=1, col=2)
        fig.update_layout(template=TMPL, height=340, margin=dict(t=40, b=10))
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, key="couv_pays")

        tg_obs = cov[cov["country_code"] == FOCUS]["observations"].iloc[0] if FOCUS in cov["country_code"].values else 0
        st.markdown(f"""
        <div class="insight">
            <strong>Interpretation</strong> — Le Togo dispose de <span class="val">{tg_obs} observations</span>
            sur {n_ind} indicateurs. L'entrainement du modele IA utilise les donnees de
            <strong>l'ensemble des 8 pays UEMOA</strong> ({n_raw:,} obs. au total), ce qui augmente
            la robustesse du modele par rapport a un entrainement uniquement sur le Togo
            ({tg_obs} obs.). C'est le principe du <strong>transfer learning regional</strong>.
        </div>""", unsafe_allow_html=True)

        st.divider()

        # --- Graph 3: Evolution temporelle d'un indicateur Togo ---
        st.markdown('<div class="sec">Explorer un indicateur — Togo</div>',
                    unsafe_allow_html=True)
        if raw_tg is not None and not raw_tg.empty:
            raw_tg_f = raw_tg[raw_tg["year"].between(*yr)]
            ind_codes = sorted(raw_tg_f["indicator_code"].unique().tolist())
            sel_ind = st.selectbox("Indicateur", ind_codes,
                                   format_func=lambda c: IND_LABELS.get(c, c),
                                   key="raw_ind")
            ri = raw_tg_f[raw_tg_f["indicator_code"] == sel_ind].sort_values("year")

            fig = go.Figure()
            ind_color = CAT_COLORS.get(IND_CAT.get(sel_ind, ""), C["accent"])
            fig.add_trace(go.Scatter(
                x=ri["year"], y=ri["value"], mode="lines+markers",
                line=dict(color=ind_color, width=2.5),
                marker=dict(size=5), fill="tozeroy",
                fillcolor=ind_color.replace(")", ",0.06)").replace("rgb", "rgba") if "rgb" in ind_color
                          else f"rgba({int(ind_color[1:3],16)},{int(ind_color[3:5],16)},{int(ind_color[5:7],16)},0.06)",
                name=IND_LABELS.get(sel_ind, sel_ind),
            ))
            fig.update_layout(
                title=f"{IND_LABELS.get(sel_ind, sel_ind)} — Togo ({yr[0]}-{yr[1]})",
                template=TMPL, height=340, margin=dict(t=40),
                hovermode="x unified", yaxis_title=IND_LABELS.get(sel_ind, ""),
            )
            st.plotly_chart(fig, key="raw_chart")

            # Auto-interpretation
            if len(ri) > 1:
                v_first, v_last = ri["value"].iloc[0], ri["value"].iloc[-1]
                chg = ((v_last / v_first) - 1) * 100 if v_first != 0 else 0
                trend = "hausse" if chg > 5 else "baisse" if chg < -5 else "stagnation"
                css = "up" if chg > 0 else "warn"
                st.markdown(f"""
                <div class="insight">
                    <strong>Lecture</strong> — <strong>{IND_LABELS.get(sel_ind, sel_ind)}</strong> au Togo
                    passe de <span class="val">{v_first:,.1f}</span> ({yr[0]})
                    a <span class="val">{v_last:,.1f}</span> ({yr[1]}),
                    soit une <span class="{css}">{trend} de {abs(chg):.1f}%</span>.
                </div>""", unsafe_allow_html=True)

        with st.expander("Voir les donnees brutes", expanded=False):
            if raw_tg is not None:
                st.dataframe(raw_tg[raw_tg["year"].between(*yr)], height=300)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORATION  (tendances + correlations)
# ═════════════════════════════════════════════════════════════════════════════
with t2:
    n_cols = len(df_all.columns)
    st.markdown(f"""
    <div class="step-box">
        <div class="step-n">Etape 2 — Transformation et exploration</div>
        <div class="step-t">Analyse des tendances historiques — Togo</div>
        <div class="step-d">Donnees pivotees, nettoyees et enrichies : {n_cols} variables
                            (21 brutes + {n_cols - 24} features d'ingenierie).</div>
    </div>""", unsafe_allow_html=True)

    if not tg.empty:
        # --- Graph 1: Population + GWh (double axe) ---
        st.markdown('<div class="sec">Croissance demographique et demande electrique</div>',
                    unsafe_allow_html=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if "SP.POP.TOTL" in tg.columns:
            fig.add_trace(go.Bar(
                x=tg["year"], y=tg["SP.POP.TOTL"] / 1e6, name="Population (M)",
                marker_color=C["pop"], opacity=0.3,
            ), secondary_y=False)
        if "conso_totale_gwh" in tg.columns:
            fig.add_trace(go.Scatter(
                x=tg["year"], y=tg["conso_totale_gwh"], name="Demande (GWh)",
                mode="lines+markers", line=dict(color=C["gwh"], width=3),
                marker=dict(size=5),
            ), secondary_y=True)
        fig.update_layout(template=TMPL, height=400, hovermode="x unified",
                          margin=dict(t=40),
                          title="Togo — Co-evolution population et demande electrique",
                          legend=dict(orientation="h", y=-0.13, font_size=10))
        fig.update_yaxes(title_text="Population (millions)", secondary_y=False)
        fig.update_yaxes(title_text="GWh", secondary_y=True)
        st.plotly_chart(fig, key="exp_pop_gwh")

        if "SP.POP.TOTL" in tg.columns and "conso_totale_gwh" in tg.columns:
            pop_chg = (tg["SP.POP.TOTL"].iloc[-1] / tg["SP.POP.TOTL"].iloc[0] - 1) * 100
            gwh_chg = (tg["conso_totale_gwh"].iloc[-1] / tg["conso_totale_gwh"].iloc[0] - 1) * 100 \
                if tg["conso_totale_gwh"].iloc[0] > 0 else 0
            elast = gwh_chg / pop_chg if pop_chg > 0 else 0
            corr = tg["SP.POP.TOTL"].corr(tg["conso_totale_gwh"])
            st.markdown(f"""
            <div class="insight">
                <strong>Analyse</strong> — Entre {yr[0]} et {yr[1]}, la population togolaise a augmente
                de <span class="up">+{pop_chg:.0f}%</span> tandis que la demande electrique a bondi
                de <span class="val">+{gwh_chg:.0f}%</span>. L'elasticite de <span class="val">{elast:.2f}</span>
                signifie que pour chaque <strong>+1%</strong> de croissance demographique,
                la demande electrique augmente de <strong>+{elast:.2f}%</strong>.
                La correlation de Pearson (<span class="val">{corr:.3f}</span>) confirme un lien
                tres fort entre population et consommation, ce qui justifie notre approche
                predictive basee sur la demographie.
            </div>""", unsafe_allow_html=True)

        st.divider()

        # --- Graph 2: Acces electricite (urbain, rural, gap) ---
        st.markdown('<div class="sec">Fracture energetique : acces urbain vs rural</div>',
                    unsafe_allow_html=True)
        if "EG.ELC.ACCS.UR.ZS" in tg.columns and "EG.ELC.ACCS.RU.ZS" in tg.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=tg["year"], y=tg["EG.ELC.ACCS.UR.ZS"], name="Urbain",
                mode="lines+markers", line=dict(color=C["good"], width=2.5),
                marker=dict(size=4),
            ))
            fig.add_trace(go.Scatter(
                x=tg["year"], y=tg["EG.ELC.ACCS.RU.ZS"], name="Rural",
                mode="lines+markers", fill="tonexty",
                fillcolor="rgba(231,76,60,0.08)",
                line=dict(color=C["warn"], width=2.5), marker=dict(size=4),
            ))
            if "EG.ELC.ACCS.ZS" in tg.columns:
                fig.add_trace(go.Scatter(
                    x=tg["year"], y=tg["EG.ELC.ACCS.ZS"], name="Moyenne nationale",
                    mode="lines", line=dict(color=C["pop"], width=2, dash="dash"),
                ))
            fig.update_layout(
                title="Acces a l'electricite — Togo",
                template=TMPL, height=360, hovermode="x unified",
                margin=dict(t=40), yaxis_title="% de la population",
                legend=dict(orientation="h", y=-0.15, font_size=10),
            )
            st.plotly_chart(fig, key="exp_acces")

            gap_now = tg["EG.ELC.ACCS.UR.ZS"].iloc[-1] - tg["EG.ELC.ACCS.RU.ZS"].iloc[-1]
            gap_old = tg["EG.ELC.ACCS.UR.ZS"].iloc[0] - tg["EG.ELC.ACCS.RU.ZS"].iloc[0]
            urb_now = tg["EG.ELC.ACCS.UR.ZS"].iloc[-1]
            rur_now = tg["EG.ELC.ACCS.RU.ZS"].iloc[-1]
            gap_dir = "se reduit" if gap_now < gap_old else "se creuse"
            st.markdown(f"""
            <div class="insight">
                <strong>Constat</strong> — La zone <strong>rurale</strong> a un taux d'acces a
                l'electricite de <span class="warn">{rur_now:.1f}%</span>, contre
                <span class="up">{urb_now:.1f}%</span> en zone urbaine,
                soit un ecart de <span class="val">{gap_now:.1f} points</span>.
                Cet ecart <strong>{gap_dir}</strong> par rapport a {yr[0]} (ecart initial de {gap_old:.1f} pts).
                Cette fracture energetique est un levier cle : chaque point de
                pourcentage d'acces gagne en zone rurale genere une hausse
                significative de la demande globale en electricite.
            </div>""", unsafe_allow_html=True)

        st.divider()

        # --- Graph 3: Contexte socio-economique (4 mini-graphiques) ---
        st.markdown('<div class="sec">Contexte socio-economique</div>',
                    unsafe_allow_html=True)

        indicators_ctx = [
            ("NY.GDP.PCAP.CD", "PIB par habitant (USD)", C["eco"]),
            ("IT.CEL.SETS.P2", "Abonnements mobile (/100 hab)", C["soc"]),
            ("SP.URB.TOTL.IN.ZS", "Urbanisation (%)", C["pop"]),
            ("SP.DYN.LE00.IN", "Esperance de vie (ans)", C["good"]),
        ]
        avail_ctx = [(code, lbl, clr) for code, lbl, clr in indicators_ctx if code in tg.columns]

        if avail_ctx:
            cols_ctx = st.columns(len(avail_ctx))
            for i, (code, lbl, clr) in enumerate(avail_ctx):
                with cols_ctx[i]:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=tg["year"], y=tg[code], mode="lines",
                        line=dict(color=clr, width=2), fill="tozeroy",
                        fillcolor=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.06)",
                    ))
                    fig.update_layout(
                        title=dict(text=lbl, font_size=11),
                        template=TMPL, height=220,
                        margin=dict(t=30, b=5, l=5, r=5),
                        xaxis=dict(showticklabels=False),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, key=f"ctx_{code}")

            st.markdown(f"""
            <div class="insight">
                <strong>Synthese</strong> — Le Togo connait une modernisation rapide :
                le PIB par habitant progresse, l'urbanisation s'accelere
                (moteur structurel de la demande electrique), et le taux de
                penetration mobile explose, signe d'une societe qui se numerise.
                Ces facteurs convergent vers une <strong>augmentation durable de la
                demande en electricite</strong>.
            </div>""", unsafe_allow_html=True)

        st.divider()

        # --- Graph 4: Matrice de correlation ---
        st.markdown('<div class="sec">Correlations entre variables cles</div>',
                    unsafe_allow_html=True)
        corr_cols_codes = ["SP.POP.TOTL", "SP.URB.TOTL.IN.ZS", "EG.ELC.ACCS.ZS",
                           "NY.GDP.PCAP.CD", "IT.CEL.SETS.P2", "SP.DYN.LE00.IN",
                           "conso_totale_gwh"]
        corr_cols = [c for c in corr_cols_codes if c in tg.columns]
        corr_labels = {
            "SP.POP.TOTL": "Population", "SP.URB.TOTL.IN.ZS": "Urbanisation",
            "EG.ELC.ACCS.ZS": "Acces elect.", "NY.GDP.PCAP.CD": "PIB/hab",
            "IT.CEL.SETS.P2": "Mobile", "SP.DYN.LE00.IN": "Esp. vie",
            "conso_totale_gwh": "Demande GWh",
        }
        if len(corr_cols) > 3:
            corr_matrix = tg[corr_cols].corr()
            labels = [corr_labels.get(c, c) for c in corr_cols]
            fig = go.Figure(go.Heatmap(
                z=corr_matrix.values, x=labels, y=labels,
                colorscale="RdBu_r", zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text:.2f}", textfont_size=10,
                colorbar=dict(thickness=12, len=0.6),
            ))
            fig.update_layout(
                title="Matrice de correlation — Variables cles (Togo)",
                template=TMPL, height=420, margin=dict(t=40),
            )
            st.plotly_chart(fig, key="corr_heatmap")

            # Identify strongest correlator with target
            target_col = "conso_totale_gwh"
            if target_col in corr_cols:
                idx = corr_cols.index(target_col)
                corr_vals = pd.Series(corr_matrix.values[idx], index=labels)
                corr_vals = corr_vals.drop(corr_labels.get(target_col, target_col), errors="ignore")
                top1_name = corr_vals.abs().idxmax()
                top1_v = corr_vals.abs().max()
                st.markdown(f"""
                <div class="insight">
                    <strong>Lecture</strong> — La variable la plus correlee a la demande electrique est
                    <strong>{top1_name}</strong> (|r| = <span class="val">{top1_v:.2f}</span>).
                    Les fortes correlations entre population, urbanisation et PIB confirment le
                    caractere <strong>structurel</strong> de la croissance de la demande : elle est
                    portee par des tendances demographiques et economiques de long terme,
                    et non par des fluctuations conjoncturelles.
                </div>""", unsafe_allow_html=True)

        with st.expander("Voir les donnees transformees", expanded=False):
            st.dataframe(tg, height=300)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODELE IA
# ═════════════════════════════════════════════════════════════════════════════
with t3:
    n_feat = len([c for c in df_all.columns if c not in ['country_code', 'country_name', 'year',
                  'conso_totale_gwh', 'EG.USE.ELEC.KH.PC']
                  and not c.startswith('EG.USE.ELEC.KH.PC')
                  and df_all[c].dtype in ['float64', 'int64', 'float32', 'int32']])
    st.markdown(f"""
    <div class="step-box">
        <div class="step-n">Etape 3 — Entrainement du modele</div>
        <div class="step-t">4 algorithmes de Machine Learning compares</div>
        <div class="step-d">{len(df_all)} observations (8 pays UEMOA), {n_feat} features,
                            variable cible : demande totale en GWh.
                            Split temporel (80% train / 20% test).</div>
    </div>""", unsafe_allow_html=True)

    # --- Graph 1: Comparaison des modeles (R2 + MAPE) ---
    if res is not None and not res.empty:
        st.markdown('<div class="sec">Performance comparee des algorithmes</div>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            r2_s = res.sort_values("r2", ascending=True)
            colors_r2 = [C["good"] if v > 0.85 else C["kwh"] if v > 0.7 else C["warn"]
                         for v in r2_s["r2"]]
            fig = go.Figure(go.Bar(
                x=r2_s["r2"], y=r2_s["model"], orientation="h",
                marker_color=colors_r2,
                text=[f"{v:.3f}" for v in r2_s["r2"]],
                textposition="outside", textfont_size=12,
            ))
            fig.update_layout(title="Score R2 (plus haut = meilleur)",
                              template=TMPL, height=300, margin=dict(t=40, l=10, r=40),
                              xaxis_range=[0, 1.05], xaxis_title="R2")
            st.plotly_chart(fig, key="mod_r2")

        with c2:
            mape_s = res.sort_values("mape", ascending=False)
            colors_mape = [C["good"] if v < 30 else C["kwh"] if v < 40 else C["warn"]
                           for v in mape_s["mape"]]
            fig = go.Figure(go.Bar(
                x=mape_s["mape"], y=mape_s["model"], orientation="h",
                marker_color=colors_mape,
                text=[f"{v:.1f}%" for v in mape_s["mape"]],
                textposition="outside", textfont_size=12,
            ))
            fig.update_layout(title="Erreur moyenne MAPE (plus bas = meilleur)",
                              template=TMPL, height=300, margin=dict(t=40, l=10, r=40),
                              xaxis_title="MAPE (%)")
            st.plotly_chart(fig, key="mod_mape")

        best = res.sort_values("r2", ascending=False).iloc[0]
        worst = res.sort_values("r2", ascending=True).iloc[0]
        st.markdown(f"""
        <div class="insight">
            <strong>Verdict</strong> — Le modele <strong>{best['model']}</strong> domine avec un
            <span class="val">R2 de {best['r2']:.3f}</span>, ce qui signifie qu'il explique
            {best['r2']*100:.1f}% de la variance de la demande electrique. Il surpasse
            significativement le {worst['model']} (R2 = {worst['r2']:.3f}).
            Le Stacking combine les forces de Random Forest, Gradient Boosting et LightGBM
            grace a un meta-modele (Ridge), ce qui reduit le biais et la variance simultanement.
        </div>""", unsafe_allow_html=True)

        # --- Graph 2: Radar des metriques ---
        st.divider()
        st.markdown('<div class="sec">Profil de performance multi-criteres</div>',
                    unsafe_allow_html=True)

        categories = ["R2", "1 - MAPE", "1 - MAE_norm", "1 - RMSE_norm"]
        max_rmse = res["rmse"].max()
        max_mae = res["mae"].max()

        fig = go.Figure()
        for _, row in res.iterrows():
            vals = [
                row["r2"],
                1 - row["mape"] / 100,
                1 - row["mae"] / max_mae,
                1 - row["rmse"] / max_rmse,
            ]
            vals.append(vals[0])  # close
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=categories + [categories[0]],
                name=row["model"], fill="toself", opacity=0.25,
            ))
        fig.update_layout(
            title="Radar — Performances multi-critere",
            template=TMPL, height=400, margin=dict(t=50),
            polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
            legend=dict(orientation="h", y=-0.1, font_size=10),
        )
        st.plotly_chart(fig, key="mod_radar")

        st.markdown(f"""
        <div class="insight">
            <strong>Lecture du radar</strong> — Plus la surface est grande, meilleur est le modele.
            Le <strong>{best['model']}</strong> occupe la plus grande surface sur l'ensemble des
            criteres (R2, erreur moyenne, erreur quadratique). C'est ce modele qui est retenu
            pour les projections jusqu'en 2045.
        </div>""", unsafe_allow_html=True)

    # --- Graph 3: Validation — observe vs predit ---
    if pred is not None and not pred.empty:
        st.divider()
        st.markdown('<div class="sec">Validation : observe vs predit — Togo</div>',
                    unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred["year"], y=pred["actual"], name="Observe",
            mode="lines+markers", line=dict(color=C["gwh"], width=2.5),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=pred["year"], y=pred["predicted"], name="Predit (IA)",
            mode="lines+markers", line=dict(color=C["proj"], width=2, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
        ))
        # Error bars
        fig.add_trace(go.Bar(
            x=pred["year"], y=pred["error"].abs(), name="Ecart absolu",
            marker_color=C["warn"], opacity=0.2,
        ))
        fig.update_layout(
            title="Togo — Validation du modele IA (observe vs predit)",
            yaxis_title="GWh", template=TMPL, height=400,
            hovermode="x unified", margin=dict(t=40),
            legend=dict(orientation="h", y=-0.13, font_size=10),
        )
        st.plotly_chart(fig, key="mod_valid")

        mae = pred["error"].abs().mean()
        rmse_val = np.sqrt((pred["error"] ** 2).mean())
        mape_val = pred["error_pct"].abs().mean() if "error_pct" in pred.columns else 0
        st.markdown(f"""
        <div class="insight">
            <strong>Validation</strong> — Sur les donnees historiques du Togo, le modele reproduit
            fidelement la trajectoire reelle avec un ecart moyen de
            <span class="val">{mae:.0f} GWh</span> (MAE). L'erreur quadratique RMSE est de
            <span class="val">{rmse_val:.0f} GWh</span>. Les barres rouges montrent l'ecart
            absolu annee par annee : on observe que le modele capture bien les
            <strong>tendances de long terme</strong>, meme si certaines annees
            presentent des deviations ponctuelles.
        </div>""", unsafe_allow_html=True)

        # Scatter real vs predicted
        st.divider()
        st.markdown('<div class="sec">Dispersion : observe vs predit</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred["actual"], y=pred["predicted"], mode="markers",
            marker=dict(color=C["gwh"], size=8, line=dict(width=1, color="#fff")),
            name="Observations",
        ))
        # Ligne parfaite
        min_v = min(pred["actual"].min(), pred["predicted"].min())
        max_v = max(pred["actual"].max(), pred["predicted"].max())
        fig.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v], mode="lines",
            line=dict(color=C["muted"], width=1, dash="dash"),
            name="Prediction parfaite",
        ))
        fig.update_layout(
            title="Diagramme de dispersion — Qualite du modele",
            xaxis_title="Observe (GWh)", yaxis_title="Predit (GWh)",
            template=TMPL, height=380, margin=dict(t=40),
            legend=dict(orientation="h", y=-0.15, font_size=10),
        )
        st.plotly_chart(fig, key="mod_scatter")

        st.markdown("""
        <div class="insight">
            <strong>Lecture</strong> — Plus les points sont proches de la diagonale pointillee,
            plus le modele est precis. On observe un <strong>alignement net</strong>,
            ce qui confirme la capacite du modele a estimer la demande reelle.
            Les points eloignes de la diagonale signalent des annees atypiques
            (chocs economiques, crises energetiques).
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTIONS 2045
# ═════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown("""
    <div class="step-box">
        <div class="step-n">Etape 4 — Prediction et projection</div>
        <div class="step-t">Anticiper la demande electrique pour le Togo de demain</div>
        <div class="step-d">Le modele IA extrapole les tendances demographiques et energetiques
                            pour projeter la demande en electricite jusqu'en 2045.</div>
    </div>""", unsafe_allow_html=True)

    if proj is not None and not proj.empty:
        proj_f = proj[proj["year"] <= proj_yr].sort_values("year") if proj_yr else proj.sort_values("year")

        # --- Grand graphique : historique + projections ---
        st.markdown('<div class="sec">Trajectoire historique et projections</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()

        if not tg.empty and "conso_totale_gwh" in tg.columns:
            fig.add_trace(go.Scatter(
                x=tg["year"], y=tg["conso_totale_gwh"],
                name="Historique observe", mode="lines+markers",
                line=dict(color=C["gwh"], width=3), marker=dict(size=5),
            ))

        fig.add_trace(go.Scatter(
            x=proj_f["year"], y=proj_f["predicted_gwh"],
            name="Projection IA", mode="lines+markers",
            line=dict(color=C["proj"], width=3),
            marker=dict(size=7, symbol="diamond"),
        ))

        fig.add_trace(go.Scatter(
            x=pd.concat([proj_f["year"], proj_f["year"][::-1]]),
            y=pd.concat([proj_f["ci_upper"], proj_f["ci_lower"][::-1]]),
            fill="toself", fillcolor=C["ci"],
            line=dict(color="rgba(0,0,0,0)"), name="Intervalle de confiance 95%",
        ))

        fig.add_vline(x=y_max_h + 0.5, line_dash="dot", line_color=C["muted"],
                      annotation_text="Transition historique / projection",
                      annotation_position="top left",
                      annotation_font_color=C["muted"], annotation_font_size=9)

        fig.update_layout(
            title=f"Togo — Demande electrique de {yr[0]} a {proj_yr or 2045}",
            yaxis_title="GWh", template=TMPL, height=460,
            hovermode="x unified", margin=dict(t=45),
            legend=dict(orientation="h", y=-0.1, font_size=10),
        )
        st.plotly_chart(fig, key="pred_main")

        last_hist_gwh = tg["conso_totale_gwh"].iloc[-1] if (
            not tg.empty and "conso_totale_gwh" in tg.columns) else 0
        last_proj_row = proj_f.iloc[-1]
        gr_total = ((last_proj_row["predicted_gwh"] / last_hist_gwh) - 1) * 100 if last_hist_gwh > 0 else 0
        ci_low, ci_high = last_proj_row["ci_lower"], last_proj_row["ci_upper"]

        st.markdown(f"""
        <div class="insight">
            <strong>Projection principale</strong> — La demande electrique du Togo devrait passer de
            <span class="val">{last_hist_gwh:,.0f} GWh</span> ({y_max_h}) a
            <span class="val">{last_proj_row['predicted_gwh']:,.0f} GWh</span>
            ({int(last_proj_row['year'])}),
            soit une hausse de <span class="up">+{gr_total:.0f}%</span>.
            L'intervalle de confiance a 95% situe la demande reelle entre
            <span class="val">{ci_low:,.0f}</span> et <span class="val">{ci_high:,.0f} GWh</span>.
            Ces chiffres integrent la croissance demographique, l'urbanisation progressive
            et l'amelioration de l'acces a l'electricite.
        </div>""", unsafe_allow_html=True)

        st.divider()

        # --- Graph 2: Population + demande projetees ---
        last_pop = tg["SP.POP.TOTL"].iloc[-1] if "SP.POP.TOTL" in tg.columns else 0
        if "pop_projected" in proj_f.columns and proj_f["pop_projected"].notna().any():
            st.markdown('<div class="sec">Demographie et demande futures</div>',
                        unsafe_allow_html=True)

            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            if "SP.POP.TOTL" in tg.columns:
                fig2.add_trace(go.Bar(
                    x=tg["year"], y=tg["SP.POP.TOTL"] / 1e6, name="Pop. historique (M)",
                    marker_color=C["pop"], opacity=0.3,
                ), secondary_y=False)
            fig2.add_trace(go.Bar(
                x=proj_f["year"], y=proj_f["pop_projected"] / 1e6, name="Pop. projetee (M)",
                marker_color=C["proj"], opacity=0.4,
            ), secondary_y=False)
            fig2.add_trace(go.Scatter(
                x=proj_f["year"], y=proj_f["predicted_gwh"], name="Demande projetee (GWh)",
                mode="lines+markers", line=dict(color=C["gwh"], width=2.5),
                marker=dict(size=4),
            ), secondary_y=True)
            fig2.update_layout(
                title="Co-evolution : population et demande projetees",
                template=TMPL, height=380, hovermode="x unified", margin=dict(t=40),
                legend=dict(orientation="h", y=-0.13, font_size=10),
            )
            fig2.update_yaxes(title_text="Population (millions)", secondary_y=False)
            fig2.update_yaxes(title_text="GWh", secondary_y=True)
            st.plotly_chart(fig2, key="pred_pop")

            pop_proj_last = last_proj_row["pop_projected"]
            gr_pop = ((pop_proj_last / last_pop) - 1) * 100 if last_pop > 0 else 0
            kwh_proj = last_proj_row["predicted_gwh"] * 1e6 / pop_proj_last if pop_proj_last > 0 else 0

            st.markdown(f"""
            <div class="insight">
                <strong>Scenario demographique</strong> — La population togolaise atteindrait
                <span class="val">{pop_proj_last/1e6:.1f} millions</span> d'habitants en
                {int(last_proj_row['year'])}, soit <span class="up">+{gr_pop:.0f}%</span>
                par rapport a {y_max_h}. Avec la demande projetee de
                <span class="val">{last_proj_row['predicted_gwh']:,.0f} GWh</span>,
                la consommation par habitant serait d'environ
                <span class="val">{kwh_proj:.0f} kWh</span>/an — un niveau qui
                reste bien en dessous de la moyenne africaine (~600 kWh),
                ce qui suggere un potentiel de croissance encore plus eleve si
                l'acces a l'electricite s'accelere.
            </div>""", unsafe_allow_html=True)

        st.divider()

        # --- Croissance annuelle projetee ---
        st.markdown('<div class="sec">Taux de croissance annuel projete de la demande</div>',
                    unsafe_allow_html=True)

        proj_growth = proj_f.copy()
        proj_growth["growth_pct"] = proj_growth["predicted_gwh"].pct_change() * 100
        proj_growth = proj_growth.dropna(subset=["growth_pct"])
        if not proj_growth.empty:
            colors_growth = [C["good"] if v > 0 else C["warn"] for v in proj_growth["growth_pct"]]
            fig = go.Figure(go.Bar(
                x=proj_growth["year"], y=proj_growth["growth_pct"],
                marker_color=colors_growth,
                text=[f"{v:+.1f}%" for v in proj_growth["growth_pct"]],
                textposition="outside", textfont_size=9,
            ))
            fig.update_layout(
                title="Croissance annuelle de la demande projetee (%)",
                yaxis_title="Croissance (%)", template=TMPL, height=320,
                margin=dict(t=40), hovermode="x unified",
            )
            fig.add_hline(y=0, line_dash="dot", line_color=C["muted"])
            avg_growth = proj_growth["growth_pct"].mean()
            st.plotly_chart(fig, key="pred_growth")

            st.markdown(f"""
            <div class="insight">
                <strong>Dynamique</strong> — Le taux de croissance annuel moyen de la demande projetee
                est de <span class="val">{avg_growth:+.1f}%</span>. Les barres vertes indiquent les
                annees de croissance, les rouges les annees de correction.
                La trajectoire n'est pas lineaire : le modele integre des cycles
                economiques et des effets de seuil lies a l'infrastructure
                energetique.
            </div>""", unsafe_allow_html=True)

        st.divider()

        # --- Consultation par annee ---
        st.markdown('<div class="sec">Consulter une annee de projection</div>',
                    unsafe_allow_html=True)
        proj_years_sel = sorted(proj_f["year"].unique().astype(int).tolist())
        sel_proj_yr = st.select_slider("Annee", options=proj_years_sel,
                                       value=proj_years_sel[-1], key="pred_yr_sel")
        row_sel = proj_f[proj_f["year"] == sel_proj_yr]
        if not row_sel.empty:
            r = row_sel.iloc[0]
            gr_gwh = ((r["predicted_gwh"] / last_hist_gwh) - 1) * 100 if last_hist_gwh > 0 else 0
            gr_pop_sel = ((r["pop_projected"] / last_pop) - 1) * 100 if (
                pd.notna(r.get("pop_projected")) and last_pop > 0) else 0
            kwh_sel = r["predicted_gwh"] * 1e6 / r["pop_projected"] if (
                pd.notna(r.get("pop_projected")) and r["pop_projected"] > 0) else 0

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=r["predicted_gwh"],
                delta={"reference": last_hist_gwh, "relative": True,
                       "valueformat": ".0%"},
                title={"text": f"Demande electrique {sel_proj_yr} (GWh)"},
                gauge={
                    "axis": {"range": [0, proj_f["ci_upper"].max() * 1.1]},
                    "bar": {"color": C["proj"]},
                    "steps": [
                        {"range": [0, last_hist_gwh], "color": "rgba(26,188,156,0.15)"},
                        {"range": [r["ci_lower"], r["ci_upper"]], "color": "rgba(155,89,182,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": C["gwh"], "width": 3},
                        "thickness": 0.75, "value": last_hist_gwh,
                    },
                },
            ))
            fig.update_layout(template=TMPL, height=300, margin=dict(t=60, b=10))
            st.plotly_chart(fig, key="pred_gauge")

            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.markdown(f"""<div class="card" style="border-left-color:{C['pop']};">
                    <div class="t">Population {sel_proj_yr}</div>
                    <div class="v">{r['pop_projected']/1e6:.1f} M</div>
                    <div class="d up">+{gr_pop_sel:.0f}% vs {y_max_h}</div></div>""",
                    unsafe_allow_html=True)
            with cc2:
                st.markdown(f"""<div class="card" style="border-left-color:{C['kwh']};">
                    <div class="t">kWh / habitant</div>
                    <div class="v">{kwh_sel:.0f} kWh</div>
                    <div class="ctx">Demande / Population</div></div>""",
                    unsafe_allow_html=True)
            with cc3:
                st.markdown(f"""<div class="card" style="border-left-color:{C['good']};">
                    <div class="t">Intervalle de confiance</div>
                    <div class="v">{r['ci_lower']:,.0f} — {r['ci_upper']:,.0f}</div>
                    <div class="ctx">GWh (IC 95%)</div></div>""",
                    unsafe_allow_html=True)

            st.markdown(f"""
            <div class="insight">
                <strong>Fiche {sel_proj_yr}</strong> — Pour une population projetee de
                <span class="val">{r['pop_projected']/1e6:.1f} M</span> habitants,
                le Togo aura besoin de <span class="val">{r['predicted_gwh']:,.0f} GWh</span>
                d'electricite, soit <span class="up">+{gr_gwh:.0f}%</span> par rapport au
                dernier point historique ({y_max_h}). La consommation par habitant s'eleverait
                a <span class="val">{kwh_sel:.0f} kWh/an</span>.
                Pour repondre a cette demande, il faudra investir dans de nouvelles
                capacites de production, de transport et de distribution electrique.
            </div>""", unsafe_allow_html=True)

    else:
        st.info("Executez python src/models/predict.py pour generer les projections.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="foot">
Source : Banque Mondiale (WDI)  |  Modeles : scikit-learn, XGBoost, LightGBM  |
Pipeline ETL Python  |  Streamlit + Plotly  |  Projet BCEAO — Developpeur IA
</div>
""", unsafe_allow_html=True)
