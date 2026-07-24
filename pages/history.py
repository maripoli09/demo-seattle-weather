import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client

st.set_page_config(page_title="Historico de Simulacoes", layout="wide")
st.title("Historico de Simulacoes")
st.caption(
    "Compara cenários guardados e identifica rapidamente as combinações com maior poupança estimada."
)

def get_supabase_client(authenticated: bool = False) -> Client | None:
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")

    if not url or not key:
        return None

    supabase = create_client(url, key)

    if authenticated:
        access_token = st.session_state.get("access_token")
        if not access_token:
            return None
        supabase.postgrest.auth(access_token)

    return supabase

def fetch_simulations(limit: int = 200, client_id: str | None = None):
    supabase = get_supabase_client(authenticated=True)

    if supabase is None:
        return None, "Sessão inválida ou Supabase não configurado."

    try:
        query = supabase.table("simulations").select("*")
        if client_id:
            query = query.eq("client_id", client_id)

        res = (
            query
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        data = res.data if hasattr(res, "data") else []
        return data, None
    
    except Exception as e:
        return None, f"Erro ao ler histórico: {e}"

if "user" not in st.session_state or st.session_state.user is None:
    st.warning("Para consultar o histórico, inicia sessão na página principal.")
    st.stop()

current_user = st.session_state.user

limit = st.slider("Número de simulações", min_value=10, max_value=200, value=50, step=10)

data, error = fetch_simulations(limit=limit, client_id=current_user.id)

if error:
    st.warning(error)
    st.stop()


if not data:
    st.info(
        "Ainda não tens cenários guardados. Cria uma simulação na página principal e clica em "
        "**Guardar cenário** para começar a comparar resultados."
    )
    st.stop()

df = pd.DataFrame(data)

numeric_columns = [
    "num_solar_panels",
    "panel_wattage",
    "predicted_consumption",
    "predicted_production",
    "energy_balance",
    "price_now",
    "estimated_cost_without_solar",
    "estimated_cost_with_solar",
    "estimated_savings",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["Data da simulação"] = df["created_at"].dt.strftime("%d/%m/%Y %H:%M")

# Se ainda não adicionaste scenario_name à tabela, cria um nome automático.
if "scenario_name" not in df.columns:
    df["scenario_name"] = (
        df["city"].fillna("Cidade não definida")
        + " — "
        + df["cycle"].fillna("Tarifa não definida")
        + " — "
        + df["num_solar_panels"].fillna(0).astype(int).astype(str)
        + " painéis"
    )

df["scenario_name"] = df["scenario_name"].fillna("Cenário sem nome")

# Filtros
st.subheader("Filtros de comparação")

filter_col_1, filter_col_2, filter_col_3 = st.columns(3)

with filter_col_1:
    cities = sorted(df["city"].dropna().unique().tolist())
    selected_cities = st.multiselect(
        "Cidade",
        options=cities,
        default=cities,
    )

with filter_col_2:
    cycles = sorted(df["cycle"].dropna().unique().tolist())
    selected_cycles = st.multiselect(
        "Ciclo tarifário",
        options=cycles,
        default=cycles,
    )

with filter_col_3:
    price_types = sorted(df["price_type"].dropna().unique().tolist())
    selected_price_types = st.multiselect(
        "Modelo de preço",
        options=price_types,
        default=price_types,
    )

df_filtered = df.copy()

if selected_cities:
    df_filtered = df_filtered[df_filtered["city"].isin(selected_cities)]

if selected_cycles:
    df_filtered = df_filtered[df_filtered["cycle"].isin(selected_cycles)]

if selected_price_types:
    df_filtered = df_filtered[df_filtered["price_type"].isin(selected_price_types)]

if df_filtered.empty:
    st.warning("Não há cenários para os filtros selecionados. Ajusta os filtros para veres resultados.")
    st.stop()

# Métricas principais
total_scenarios = len(df_filtered)

if "estimated_savings" in df_filtered.columns:
    savings_series = pd.to_numeric(df_filtered["estimated_savings"], errors="coerce")
else:
    savings_series = pd.Series(dtype=float)

average_savings = float(savings_series.mean()) if not savings_series.dropna().empty else 0.0

if savings_series.dropna().empty:
    best_savings = 0.0
    best_scenario_name = "Sem dados suficientes"
else:
    best_idx = savings_series.idxmax()
    best_scenario = df_filtered.loc[best_idx]
    best_savings = float(best_scenario["estimated_savings"])
    best_scenario_name = best_scenario["scenario_name"]

metric_1, metric_2, metric_3 = st.columns(3)

metric_1.metric(
    "Cenários analisados",
    total_scenarios,
)

metric_2.metric(
    "Poupança média estimada",
    f"{average_savings:.2f} €",
)

metric_3.metric(
    "Maior poupança estimada",
    f"{best_savings:.2f} €",
)

st.success(
    f"Melhor cenário atual: **{best_scenario_name}** | Poupança estimada: **{best_savings:.2f} €**."
)

st.divider()

# Gráfico 1: custos com e sem solar.
st.subheader("Custo estimado por cenário")

cost_columns = [
    "scenario_name",
    "estimated_cost_without_solar",
    "estimated_cost_with_solar",
]

if all(column in df_filtered.columns for column in cost_columns):
    df_costs = df_filtered[cost_columns].melt(
        id_vars="scenario_name",
        value_vars=[
            "estimated_cost_without_solar",
            "estimated_cost_with_solar",
        ],
        var_name="Tipo de custo",
        value_name="Custo estimado (€)",
    )

    df_costs["Tipo de custo"] = df_costs["Tipo de custo"].replace(
        {
            "estimated_cost_without_solar": "Sem produção solar",
            "estimated_cost_with_solar": "Com produção solar",
        }
    )

    fig_costs = px.bar(
        df_costs,
        x="scenario_name",
        y="Custo estimado (€)",
        color="Tipo de custo",
        barmode="group",
        labels={
            "scenario_name": "Cenário",
            "Custo estimado (€)": "Custo estimado (€)",
        },
        title="Custo estimado: com e sem autoconsumo solar",
        color_discrete_map={
            "Sem produção solar": "#E76F51",
            "Com produção solar": "#2A9D8F",
        },
    )

    st.plotly_chart(fig_costs, use_container_width=True)

st.divider()

# Gráfico 2: poupança por cenário.
st.subheader("Ranking de poupança estimada")

if "estimated_savings" in df_filtered.columns:
    df_savings = df_filtered.sort_values(
        "estimated_savings",
        ascending=False,
    )

    fig_savings = px.bar(
        df_savings,
        x="scenario_name",
        y="estimated_savings",
        color="cycle",
        text="estimated_savings",
        labels={
            "scenario_name": "Cenário",
            "estimated_savings": "Poupança estimada (€)",
            "cycle": "Ciclo tarifário",
        },
        title="Cenários ordenados por maior poupança",
    )

    fig_savings.update_traces(
        texttemplate="%{text:.2f} €",
        textposition="outside",
    )

    st.plotly_chart(fig_savings, use_container_width=True)

st.divider()

# Gráfico 3: impacto do número de painéis solares.
st.subheader("Impacto do número de painéis na poupança")

required_columns = [
    "num_solar_panels",
    "estimated_savings",
    "scenario_name",
    "city",
]

if all(column in df_filtered.columns for column in required_columns):
    fig_panels = px.scatter(
        df_filtered,
        x="num_solar_panels",
        y="estimated_savings",
        color="city",
        size="predicted_production"
        if "predicted_production" in df_filtered.columns
        else None,
        hover_name="scenario_name",
        labels={
            "num_solar_panels": "Número de painéis solares",
            "estimated_savings": "Poupança estimada (€)",
            "city": "Cidade",
        },
        title="Relação entre número de painéis e poupança estimada",
    )

    st.plotly_chart(fig_panels, use_container_width=True)

st.divider()

# Tabela detalhada.
st.subheader("Tabela detalhada")

visible_columns = [
    "Data da simulação",
    "scenario_name",
    "city",
    "cycle",
    "price_type",
    "num_solar_panels",
    "panel_wattage",
    "predicted_consumption",
    "predicted_production",
    "energy_balance",
    "estimated_cost_without_solar",
    "estimated_cost_with_solar",
    "estimated_savings",
    "model_version",
]

visible_columns = [
    column for column in visible_columns
    if column in df_filtered.columns
]

df_display = df_filtered[visible_columns].copy()

df_display = df_display.rename(
    columns={
        "scenario_name": "Nome do cenário",
        "city": "Cidade",
        "cycle": "Ciclo tarifário",
        "price_type": "Modelo de preço",
        "num_solar_panels": "N.º de painéis",
        "panel_wattage": "Potência por painel (W)",
        "predicted_consumption": "Consumo previsto (kWh)",
        "predicted_production": "Produção solar prevista (kWh)",
        "energy_balance": "Balanço energético (kWh)",
        "estimated_cost_without_solar": "Custo sem solar (€)",
        "estimated_cost_with_solar": "Custo com solar (€)",
        "estimated_savings": "Poupança estimada (€)",
        "model_version": "Versão do modelo",
    }
)

st.dataframe(
    df_display,
    use_container_width=True,
    hide_index=True,
)

csv = df_display.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descarregar resultados (CSV)",
    data=csv,
    file_name="cenarios_energeticos.csv",
    mime="text/csv",
)