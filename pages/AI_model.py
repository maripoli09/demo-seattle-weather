import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Analise de Dados - 300 Clientes", layout="wide")
st.title("📊 Analise de Dados dos 300 Clientes")
st.caption("Amostra: 300 clientes do dataset Ausgrid")

# 1) Carregar dados
# Ajusta o caminho se o ficheiro estiver noutra pasta
DATA_PATH = Path("df_gc_clean.pkl")

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    return df

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Nao foi possivel ler o ficheiro: {e}")
    st.stop()

st.markdown(
    """
    Esta pagina analisa uma amostra de 300 clientes do dataset Ausgrid.
    A agregacao e feita por timestamp para encontrar padroes horarios, semanais e mensais de consumo.
    """
)

# 2) Normalizar nomes de colunas (ajusta conforme o teu dataframe)
# Tenta encontrar coluna de timestamp e consumo
possible_ts = ["timestamp", "Timestamp", "date", "datetime", "DateTime", "TS"]
possible_cons = ["consumo", "consumption", "kwh", "energy", "value", "GC"]

ts_col = next((c for c in possible_ts if c in df.columns), None)
cons_col = next((c for c in possible_cons if c in df.columns), None)

if ts_col is None or cons_col is None:
    st.error(
        f"Nao encontrei colunas esperadas. Colunas atuais: {list(df.columns)}. "
        "Confirma qual e a coluna de timestamp e qual e a de consumo."
    )
    st.stop()

df = df.copy()
df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
df = df.dropna(subset=[ts_col, cons_col])

# Variaveis temporais para agregacao
df["hora"] = df[ts_col].dt.hour
df["mes"] = df[ts_col].dt.month
df["dia_semana"] = df[ts_col].dt.dayofweek  # 0=segunda ... 6=domingo
df["tipo_dia"] = df["dia_semana"].apply(lambda x: "Fim de Semana" if x >= 5 else "Dia Util")

nomes_dias = {
    0: "Segunda", 1: "Terca", 2: "Quarta", 3: "Quinta",
    4: "Sexta", 5: "Sabado", 6: "Domingo"
}
df["dia_semana_nome"] = df["dia_semana"].map(nomes_dias)

# 3) Consumo medio por hora
st.subheader("Consumo Medio por Hora")
by_hour = df.groupby("hora", as_index=False)[cons_col].mean()

chart_hour = (
    alt.Chart(by_hour)
    .mark_line(point=True)
    .encode(
        x=alt.X("hora:Q", title="Hora do Dia"),
        y=alt.Y(f"{cons_col}:Q", title="Consumo Medio"),
        tooltip=["hora", alt.Tooltip(f"{cons_col}:Q", format=".4f")]
    )
    .properties(height=300)
)
st.altair_chart(chart_hour, use_container_width=True)

peak_hour = int(by_hour.loc[by_hour[cons_col].idxmax(), "hora"])
st.markdown(
    f"Padrao observado: o pico medio acontece por volta das {peak_hour}h, "
    "o que pode indicar concentracao de uso em horarios especificos."
)

# 4) Consumo medio por mes
st.subheader("Consumo Medio por Mes")
by_month = df.groupby("mes", as_index=False)[cons_col].mean()

chart_month = (
    alt.Chart(by_month)
    .mark_bar()
    .encode(
        x=alt.X("mes:O", title="Mes"),
        y=alt.Y(f"{cons_col}:Q", title="Consumo Medio"),
        tooltip=["mes", alt.Tooltip(f"{cons_col}:Q", format=".4f")]
    )
    .properties(height=300)
)
st.altair_chart(chart_month, use_container_width=True)

peak_month = int(by_month.loc[by_month[cons_col].idxmax(), "mes"])
st.markdown(
    f"Padrao observado: o mes com maior consumo medio foi {peak_month}. "
    "Compara este comportamento com sazonalidade (temperatura e uso de climatizacao)."
)

# 5) Consumo medio por dia da semana
st.subheader("Consumo Medio por Dia da Semana")
by_weekday = (
    df.groupby(["dia_semana", "dia_semana_nome"], as_index=False)[cons_col]
    .mean()
    .sort_values("dia_semana")
)

chart_weekday = (
    alt.Chart(by_weekday)
    .mark_bar()
    .encode(
        x=alt.X("dia_semana_nome:O", title="Dia da Semana"),
        y=alt.Y(f"{cons_col}:Q", title="Consumo Medio"),
        tooltip=["dia_semana_nome", alt.Tooltip(f"{cons_col}:Q", format=".4f")]
    )
    .properties(height=300)
)
st.altair_chart(chart_weekday, use_container_width=True)

top_day = by_weekday.loc[by_weekday[cons_col].idxmax(), "dia_semana_nome"]
st.markdown(
    f"Padrao observado: {top_day} apresenta o maior consumo medio entre os dias da semana."
)

# 6) Dias uteis vs fins de semana
st.subheader("Comparacao: Dias Uteis vs Fins de Semana")
by_daytype = df.groupby("tipo_dia", as_index=False)[cons_col].mean()

chart_daytype = (
    alt.Chart(by_daytype)
    .mark_bar()
    .encode(
        x=alt.X("tipo_dia:O", title="Tipo de Dia"),
        y=alt.Y(f"{cons_col}:Q", title="Consumo Medio"),
        color="tipo_dia:N",
        tooltip=["tipo_dia", alt.Tooltip(f"{cons_col}:Q", format=".4f")]
    )
    .properties(height=300)
)
st.altair_chart(chart_daytype, use_container_width=True)

util = by_daytype.loc[by_daytype["tipo_dia"] == "Dia Util", cons_col]
fim = by_daytype.loc[by_daytype["tipo_dia"] == "Fim de Semana", cons_col]
if len(util) and len(fim) and fim.iloc[0] != 0:
    diff = (util.iloc[0] - fim.iloc[0]) / fim.iloc[0] * 100
    st.markdown(
        f"Padrao observado: dias uteis estao {diff:.1f}% "
        f"{'acima' if diff >= 0 else 'abaixo'} dos fins de semana."
    )

# 7) Heatmap hora x dia da semana
st.subheader("Heatmap: Hora x Dia da Semana")
heat = (
    df.groupby(["dia_semana_nome", "hora"], as_index=False)[cons_col]
    .mean()
)

# Ordem correta dos dias
ordem_dias = ["Segunda", "Terca", "Quarta", "Quinta", "Sexta", "Sabado", "Domingo"]

chart_heat = (
    alt.Chart(heat)
    .mark_rect()
    .encode(
        x=alt.X("hora:O", title="Hora"),
        y=alt.Y("dia_semana_nome:O", sort=ordem_dias, title="Dia da Semana"),
        color=alt.Color(f"{cons_col}:Q", title="Consumo Medio"),
        tooltip=["dia_semana_nome", "hora", alt.Tooltip(f"{cons_col}:Q", format=".4f")]
    )
    .properties(height=320)
)
st.altair_chart(chart_heat, use_container_width=True)

st.markdown(
    "Padrao observado: o heatmap mostra os blocos de maior intensidade de consumo "
    "ao cruzar hora e dia da semana, facilitando identificar picos recorrentes."
)

# 8) Distribuicao por categoria (GC, GG, CL) se existir
st.subheader("Distribuicao do Consumo por Categoria (GC, GG, CL)")

category_col_candidates = ["categoria", "category", "tariff", "classe", "type"]
cat_col = next((c for c in category_col_candidates if c in df.columns), None)

if cat_col is not None:
    cats = ["GC", "GG", "CL"]
    dcat = df[df[cat_col].astype(str).isin(cats)].groupby(cat_col, as_index=False)[cons_col].mean()

    if not dcat.empty:
        chart_cat = (
            alt.Chart(dcat)
            .mark_bar()
            .encode(
                x=alt.X(f"{cat_col}:N", title="Categoria"),
                y=alt.Y(f"{cons_col}:Q", title="Consumo Medio"),
                color=f"{cat_col}:N",
                tooltip=[cat_col, alt.Tooltip(f"{cons_col}:Q", format=".4f")]
            )
            .properties(height=300)
        )
        st.altair_chart(chart_cat, use_container_width=True)
        st.markdown(
            "Padrao observado: a comparacao entre GC, GG e CL ajuda a identificar "
            "segmentos com perfis de consumo distintos."
        )
    else:
        st.info("Existe coluna de categoria, mas nao encontrei valores GC/GG/CL.")
else:
    st.info("Nao existe coluna de categoria identificada para GC/GG/CL.")