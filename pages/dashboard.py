import streamlit as st
import pandas as pd
from datetime import datetime
import altair as alt
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Analise de Dados - 300 Clientes", layout="wide")
st.title("Análise de Dados de Consumo")
st.caption("Amostra agregada de 300 clientes (dataset Ausgrid).")

DATA_PATH = Path("df_gc_clean.pkl")

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    try:
        return pd.read_pickle(path)
    except Exception:
        return joblib.load(path)

def find_column(columns, candidates):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Nao foi possivel carregar df_gc_clean.pkl: {e}")
    st.stop()

st.markdown(
    """
    Esta página mostra uma leitura rápida dos padrões de consumo ao longo do tempo.
    Usa agregação por timestamp para identificar tendências por hora, dia da semana e mês.
    """
)

st.info(
    "Usa os filtros para focar períodos e categorias. Os gráficos destacam padrões úteis para apoiar decisões tarifárias e operacionais."
)

timestamp_candidates = [
    "timestamp", "datetime", "date_time", "date", "data_hora", "ts"
]
consumption_candidates = [
    "energy_kwh", "consumo", "consumption", "kwh", "value", "gc"
]
category_candidates = [
    "consumption category", "consumption_category", "category",
    "categoria", "type", "classe"
]

ts_col = find_column(df.columns, timestamp_candidates)
cons_col = find_column(df.columns, consumption_candidates)
cat_col = find_column(df.columns, category_candidates)

if ts_col is None or cons_col is None:
    st.error(
        "Nao encontrei colunas obrigatorias de timestamp e consumo. "
        f"Colunas disponiveis: {list(df.columns)}"
    )
    st.stop()

df = df.copy()
df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
df = df.dropna(subset=[ts_col, cons_col])
df[cons_col] = pd.to_numeric(df[cons_col], errors="coerce")
df = df.dropna(subset=[cons_col])

# Filtros
with st.sidebar:
    st.header("Filtros")

    if cat_col is not None:
        categorias = sorted(df[cat_col].dropna().astype(str).unique().tolist())
        cat_sel = st.multiselect("Categoria", categorias, default=categorias)
        if cat_sel:
            df = df[df[cat_col].astype(str).isin(cat_sel)]

    min_date = df[ts_col].min().date()
    max_date = df[ts_col].max().date()
    date_range = st.date_input("Periodo", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df[ts_col].dt.date >= start_date) & (df[ts_col].dt.date <= end_date)
        df = df[mask]

if df.empty:
    st.warning("Sem dados após aplicação dos filtros. Ajusta os critérios para continuar.")
    st.stop()

# Engenharia temporal
df["hora"] = df[ts_col].dt.hour
df["mes"] = df[ts_col].dt.month
df["dia_semana"] = df[ts_col].dt.dayofweek
df["tipo_dia"] = df["dia_semana"].apply(lambda x: "Fim de Semana" if x >= 5 else "Dia Util")

nomes_dias = {
    0: "Segunda",
    1: "Terca",
    2: "Quarta",
    3: "Quinta",
    4: "Sexta",
    5: "Sabado",
    6: "Domingo",
}
df["dia_semana_nome"] = df["dia_semana"].map(nomes_dias)

nomes_meses = {
    1: "Jul", 2: "Ago", 3: "Set", 4: "Out", 5: "Nov", 6: "Dez",
    7: "Jan", 8: "Fev", 9: "Mar", 10: "Abr", 11: "Mai", 12: "Jun"
}
df["mes_nome"] = df["mes"].map(nomes_meses)

# KPIs
c1, c2, c3 = st.columns(3)
c1.metric("Registos analisados", f"{len(df):,}".replace(",", "."))
c2.metric("Consumo medio global", f"{df[cons_col].mean():.3f} kWh")
c3.metric("Desvio padrao", f"{df[cons_col].std():.3f} kWh")

st.divider()

st.subheader("Perfil energético diário completo (GC, GG e CL)")

fallback_hourly = df.groupby("hora", as_index=False)[cons_col].mean()

def _daily_profile_markers(profile: pd.DataFrame, value_col: str) -> tuple[int, int, int]:
    madrugada = profile[(profile["hora"] >= 0) & (profile["hora"] <= 6)]
    manha = profile[(profile["hora"] >= 6) & (profile["hora"] <= 12)]
    noite = profile[(profile["hora"] >= 17) & (profile["hora"] <= 23)]

    min_madrugada_h = int(
        madrugada.loc[madrugada[value_col].idxmin(), "hora"]
        if not madrugada.empty
        else profile.loc[profile[value_col].idxmin(), "hora"]
    )
    pico_manha_h = int(
        manha.loc[manha[value_col].idxmax(), "hora"]
        if not manha.empty
        else profile.loc[profile[value_col].idxmax(), "hora"]
    )
    pico_noite_h = int(
        noite.loc[noite[value_col].idxmax(), "hora"]
        if not noite.empty
        else profile.loc[profile[value_col].idxmax(), "hora"]
    )

    return min_madrugada_h, pico_manha_h, pico_noite_h

def _render_fallback_profile() -> None:
    min_h, manha_h, noite_h = _daily_profile_markers(fallback_hourly, cons_col)

    markers = pd.DataFrame(
        {
            "hora": [min_h, manha_h, noite_h],
            "tipo": [
                "Minimo madrugada",
                "Pico matinal",
                "Pico final da tarde/noite",
            ],
        }
    ).merge(fallback_hourly, on="hora", how="left")

    base = (
        alt.Chart(fallback_hourly)
        .mark_line(point=True, strokeWidth=2.8, color="#2563EB")
        .encode(
            x=alt.X("hora:Q", title="Hora do dia", scale=alt.Scale(domain=[0, 23])),
            y=alt.Y(f"{cons_col}:Q", title="Consumo medio agregado (kWh)"),
            tooltip=[
                alt.Tooltip("hora:Q", title="Hora"),
                alt.Tooltip(f"{cons_col}:Q", title="kWh", format=".4f"),
            ],
        )
    )

    highlights = (
        alt.Chart(markers)
        .mark_circle(size=140)
        .encode(
            x="hora:Q",
            y=f"{cons_col}:Q",
            color=alt.Color(
                "tipo:N",
                scale=alt.Scale(
                    domain=[
                        "Minimo madrugada",
                        "Pico matinal",
                        "Pico final da tarde/noite",
                    ],
                    range=["#14B8A6", "#F59E0B", "#EF4444"],
                ),
                title="Pontos de leitura",
            ),
            tooltip=[
                alt.Tooltip("tipo:N", title="Leitura"),
                alt.Tooltip("hora:Q", title="Hora"),
                alt.Tooltip(f"{cons_col}:Q", title="kWh", format=".4f"),
            ],
        )
    )

    st.altair_chart((base + highlights).properties(height=360), use_container_width=True)
    st.markdown(
        f"""
        Leitura automática no consumo agregado:

        - minimo de madrugada por volta das **{min_h:02d}h**;
        - pico matinal por volta das **{manha_h:02d}h**;
        - pico de final de tarde/noite por volta das **{noite_h:02d}h**.
        """
    )

if cat_col is not None:
    profile_df = df[["hora", cat_col, cons_col]].copy()
    profile_df["_cat_norm"] = (
        profile_df[cat_col]
        .astype(str)
        .str.upper()
        .str.extract(r"(GC|GG|CL)", expand=False)
    )

    label_map = {
        "GC": "Consumo Geral (GC)",
        "GG": "Producao Solar (GG)",
        "CL": "Cargas Controladas (CL)",
    }
    color_map = {
        "Consumo Geral (GC)": "#2563EB",
        "Producao Solar (GG)": "#F59E0B",
        "Cargas Controladas (CL)": "#16A34A",
    }

    profile_df = profile_df[profile_df["_cat_norm"].isin(label_map.keys())]

    if not profile_df.empty:
        profile_hourly = (
            profile_df.groupby(["hora", "_cat_norm"], as_index=False)[cons_col]
            .mean()
        )
        profile_hourly["Serie"] = profile_hourly["_cat_norm"].map(label_map)

        chart_profile = (
            alt.Chart(profile_hourly)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("hora:Q", title="Hora do dia", scale=alt.Scale(domain=[0, 23])),
                y=alt.Y(f"{cons_col}:Q", title="Energia media (kWh)"),
                color=alt.Color(
                    "Serie:N",
                    scale=alt.Scale(
                        domain=list(color_map.keys()),
                        range=list(color_map.values()),
                    ),
                    title="Serie",
                ),
                tooltip=[
                    alt.Tooltip("hora:Q", title="Hora"),
                    alt.Tooltip("Serie:N", title="Serie"),
                    alt.Tooltip(f"{cons_col}:Q", title="kWh", format=".4f"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(chart_profile, use_container_width=True)

        gc_profile = profile_hourly[profile_hourly["_cat_norm"] == "GC"]
        if not gc_profile.empty:
            min_madrugada_h, pico_manha_h, pico_noite_h = _daily_profile_markers(gc_profile, cons_col)

            st.markdown(
                f"""
                Leituras principais no consumo agregado (GC):

                - minimo de madrugada por volta das **{min_madrugada_h:02d}h**;
                - pico matinal por volta das **{pico_manha_h:02d}h**;
                - pico de final de tarde/noite por volta das **{pico_noite_h:02d}h**.
                """
            )
    else:
        st.info("Nao foram encontradas categorias GC/GG/CL nos dados filtrados. Abaixo segue o perfil agregado disponivel.")
        _render_fallback_profile()
else:
    st.info(
        "Este dataset da app (df_gc_clean.pkl) representa consumo agregado e nao inclui categorias GC/GG/CL. "
        "Abaixo e mostrado o perfil diario agregado, com os principais pontos de leitura."
    )
    _render_fallback_profile()

# 1) Consumo medio por hora
st.subheader("1) Consumo médio por hora")
by_hour = df.groupby("hora", as_index=False)[cons_col].mean()
peak_hour = int(by_hour.loc[by_hour[cons_col].idxmax(), "hora"])

col_chart, col_text = st.columns([2, 1])
with col_chart:
    chart_hour = (
        alt.Chart(by_hour)
        .mark_line(point=True)
        .encode(
            x=alt.X("hora:Q", title="Hora"),
            y=alt.Y(f"{cons_col}:Q", title="Consumo medio (kWh)"),
            tooltip=["hora", alt.Tooltip(f"{cons_col}:Q", format=".4f")]
        )
        .properties(height=320)
    )
    st.altair_chart(chart_hour, use_container_width=True)
with col_text:
    st.markdown(
        f"""
        Leitura por hora considerando os dados filtrados.

        Insight principal:
        o pico médio surge por volta das {peak_hour}h, sugerindo concentração de uso nesse período.
        """
    )

# 2) Consumo medio por mes
st.subheader("2) Consumo médio por mês")
by_month = (
    df.groupby(["mes", "mes_nome"], as_index=False)[cons_col]
    .mean()
    .sort_values("mes")
)
peak_month = by_month.loc[by_month[cons_col].idxmax(), "mes_nome"]

col_chart, col_text = st.columns([2, 1])
with col_chart:
    chart_month = (
        alt.Chart(by_month)
        .mark_bar()
        .encode(
            x=alt.X("mes_nome:N", title="Mes", sort=list(nomes_meses.values())),
            y=alt.Y(f"{cons_col}:Q", title="Consumo medio (kWh)"),
            tooltip=["mes_nome", alt.Tooltip(f"{cons_col}:Q", format=".4f")]
        )
        .properties(height=320)
    )
    st.altair_chart(chart_month, use_container_width=True)
with col_text:
    st.markdown(
        f"""
        Visão mensal para observar sazonalidade.

        Insight principal:
        mês com maior média de consumo: {peak_month}.
        """
    )

# 3) Consumo medio por dia da semana
st.subheader("3) Consumo médio por dia da semana")
ordem_dias = ["Segunda", "Terca", "Quarta", "Quinta", "Sexta", "Sabado", "Domingo"]
by_weekday = (
    df.groupby(["dia_semana", "dia_semana_nome"], as_index=False)[cons_col]
    .mean()
    .sort_values("dia_semana")
)
top_day = by_weekday.loc[by_weekday[cons_col].idxmax(), "dia_semana_nome"]

col_chart, col_text = st.columns([2, 1])
with col_chart:
    chart_weekday = (
        alt.Chart(by_weekday)
        .mark_bar()
        .encode(
            x=alt.X("dia_semana_nome:N", sort=ordem_dias, title="Dia da semana"),
            y=alt.Y(f"{cons_col}:Q", title="Consumo medio (kWh)"),
            tooltip=["dia_semana_nome", alt.Tooltip(f"{cons_col}:Q", format=".4f")]
        )
        .properties(height=320)
    )
    st.altair_chart(chart_weekday, use_container_width=True)
with col_text:
    st.markdown(
        f"""
        Comparação semanal do comportamento de consumo.

        Insight principal:
        maior consumo médio em {top_day}.
        """
    )

# 4) Dias uteis vs fins de semana
st.subheader("4) Comparação: dias úteis vs fins de semana")
by_daytype = df.groupby("tipo_dia", as_index=False)[cons_col].mean()

util = by_daytype.loc[by_daytype["tipo_dia"] == "Dia Util", cons_col]
fim = by_daytype.loc[by_daytype["tipo_dia"] == "Fim de Semana", cons_col]
if len(util) and len(fim) and fim.iloc[0] != 0:
    diff_pct = ((util.iloc[0] - fim.iloc[0]) / fim.iloc[0]) * 100
else:
    diff_pct = 0.0

col_chart, col_text = st.columns([2, 1])
with col_chart:
    chart_daytype = (
        alt.Chart(by_daytype)
        .mark_bar()
        .encode(
            x=alt.X("tipo_dia:N", title="Tipo de dia"),
            y=alt.Y(f"{cons_col}:Q", title="Consumo medio (kWh)"),
            color=alt.Color("tipo_dia:N", legend=None),
            tooltip=["tipo_dia", alt.Tooltip(f"{cons_col}:Q", format=".4f")]
        )
        .properties(height=320)
    )
    st.altair_chart(chart_daytype, use_container_width=True)
with col_text:
    st.markdown(
        f"""
        Comparação direta entre os dois perfis de utilização.

        Insight principal:
        dias úteis estão {abs(diff_pct):.1f}% {"acima" if diff_pct >= 0 else "abaixo"} dos fins de semana.
        """
    )

# 5) Heatmap hora x dia da semana
st.subheader("5) Heatmap: hora x dia da semana")
heat = (
    df.groupby(["dia_semana_nome", "hora"], as_index=False)[cons_col]
    .mean()
)

col_chart, col_text = st.columns([2, 1])
with col_chart:
    chart_heat = (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("hora:O", title="Hora"),
            y=alt.Y("dia_semana_nome:N", sort=ordem_dias, title="Dia da semana"),
            color=alt.Color(f"{cons_col}:Q", title="Consumo medio"),
            tooltip=["dia_semana_nome", "hora", alt.Tooltip(f"{cons_col}:Q", format=".4f")]
        )
        .properties(height=360)
    )
    st.altair_chart(chart_heat, use_container_width=True)
with col_text:
    st.markdown(
        """
        Cruzamento de duas dimensões temporais para encontrar picos recorrentes.

        Insight principal:
        os blocos mais intensos destacam janelas horárias críticas para gestão de carga.
        """
    )

st.divider()

# 7) Heatmap hora x mes
st.subheader("6) Heatmap: hora x mês")
heat_month = (
    df.groupby(["mes_nome", "hora"], as_index=False)[cons_col]
    .mean()
)

ordem_meses = list(nomes_meses.values())

col_chart, col_text = st.columns([2, 1])
with col_chart:
    chart_heat_month = (
        alt.Chart(heat_month)
        .mark_rect()
        .encode(
            x=alt.X("hora:O", title="Hora"),
            y=alt.Y("mes_nome:N", sort=ordem_meses, title="Mes"),
            color=alt.Color(f"{cons_col}:Q", title="Consumo medio"),
            tooltip=["mes_nome", "hora", alt.Tooltip(f"{cons_col}:Q", format=".4f")]
        )
        .properties(height=360)
    )
    st.altair_chart(chart_heat_month, use_container_width=True)

with col_text:
    st.markdown(
        """
        Mostra sazonalidade intra-diária ao longo dos meses.

        Insight principal:
        permite identificar horas de pico em cada mês.
        """
    )

# 8) Outliers por IQR
st.subheader("7) Outliers por categoria (IQR)")

iqr_rows = []
if cat_col is not None:
    categorias_iqr = [("Consumo Geral", "GC"), ("Producao Solar", "GG"), ("Cargas Controladas", "CL")]

    for nome_cat, codigo_cat in categorias_iqr:
        serie = df[df[cat_col].astype(str) == codigo_cat][cons_col].dropna()
        if len(serie) == 0:
            continue

        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        up = q3 + 1.5 * iqr

        n_out = int(((serie < low) | (serie > up)).sum())
        pct_out = (n_out / len(serie)) * 100

        iqr_rows.append(
            {
                "Categoria": codigo_cat,
                "Descricao": nome_cat,
                "Q1": round(float(q1), 4),
                "Q3": round(float(q3), 4),
                "IQR": round(float(iqr), 4),
                "Limite_inf": round(float(low), 4),
                "Limite_sup": round(float(up), 4),
                "Outliers_n": n_out,
                "Outliers_%": round(float(pct_out), 2),
            }
        )

if not iqr_rows:
    fallback_groups = [("Dia Util", df[df["tipo_dia"] == "Dia Util"]), ("Fim de Semana", df[df["tipo_dia"] == "Fim de Semana"])]

    for nome_grupo, subset in fallback_groups:
        serie = subset[cons_col].dropna()
        if len(serie) == 0:
            continue

        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        up = q3 + 1.5 * iqr

        n_out = int(((serie < low) | (serie > up)).sum())
        pct_out = (n_out / len(serie)) * 100

        iqr_rows.append(
            {
                "Categoria": nome_grupo,
                "Descricao": nome_grupo,
                "Q1": round(float(q1), 4),
                "Q3": round(float(q3), 4),
                "IQR": round(float(iqr), 4),
                "Limite_inf": round(float(low), 4),
                "Limite_sup": round(float(up), 4),
                "Outliers_n": n_out,
                "Outliers_%": round(float(pct_out), 2),
            }
        )

if iqr_rows:
    st.dataframe(pd.DataFrame(iqr_rows), use_container_width=True)
else:
    st.info("Sem dados suficientes para calcular IQR nos grupos disponiveis.")


# 9) Boxplot por categoria em escala log
st.subheader("8) Distribuição por categoria (escala log)")

if cat_col is not None:
    dbox = df[df[cat_col].astype(str).isin(["GC", "GG", "CL"])].copy()
    group_col = cat_col
    group_title = "Categoria"
else:
    dbox = df.copy()
    group_col = "tipo_dia"
    group_title = "Tipo de dia"

dbox = dbox[pd.to_numeric(dbox[cons_col], errors="coerce") > 0]

if not dbox.empty:
    chart_box = (
        alt.Chart(dbox)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X(f"{group_col}:N", title=group_title),
            y=alt.Y(
                f"{cons_col}:Q",
                title="Consumo (kWh, log)",
                scale=alt.Scale(type="log")
            ),
            color=alt.Color(f"{group_col}:N", legend=None),
            tooltip=[group_col]
        )
        .properties(height=320)
    )
    st.altair_chart(chart_box, use_container_width=True)
else:
    st.info("Sem valores positivos suficientes para a escala log nos grupos disponiveis.")

st.divider()

st.subheader("Conclusoes principais")
st.markdown(
    f"""
    - A análise confirma padrões temporais claros no consumo agregado.
    - O pico horário médio ocorre em torno das {peak_hour}h.
    - O comportamento mensal indica sazonalidade.
    - A diferença entre dias úteis e fins de semana é relevante para planeamento energético.
    - Os heatmaps ajudam a identificar janelas críticas para previsão e recomendações contextuais.
    """
)