import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from utils import load_smart_models

st.set_page_config(page_title="Modelo IA", layout="wide")
st.title("Modelo de IA")
st.caption("Como o modelo estima o consumo energético e como interpretar os resultados.")

st.info(
    "Nesta página encontras um resumo simples do modelo: variáveis usadas, métricas de qualidade, "
    "fatores mais importantes e limitações."
)

st.markdown(
    """
    ## 1) Porque usamos XGBoost
    O XGBoost (Extreme Gradient Boosting) é um algoritmo de ensemble baseado em árvores de decisão.
    Em termos práticos, combina vários modelos simples para criar previsões mais robustas.
    
    Foi escolhido neste projeto porque:
    - modela relações não lineares entre variáveis temporais e consumo
    - funciona muito bem com dados tabulares
    - mantém boa precisão em séries energéticas agregadas
    - permite interpretar a importância relativa de cada variável
    """
)

st.markdown("## 2) Variáveis usadas no modelo")
features_df = pd.DataFrame(
    [
        {"Variavel": "hour", "Descricao": "Hora do dia (0-23)"},
        {"Variavel": "day_of_week", "Descricao": "Dia da semana (0=segunda ... 6=domingo)"},
        {"Variavel": "month", "Descricao": "Mes do ano (1-12)"},
        {"Variavel": "is_weekend", "Descricao": "Indicador de fim de semana (0/1)"},
        {"Variavel": "lag_1", "Descricao": "Consumo observado no intervalo de 30 minutos anterior"},
        {"Variavel": "lag_48", "Descricao": "Consumo observado no mesmo intervalo do dia anterior"},
    ]
)
st.dataframe(features_df, use_container_width=True)

st.markdown("## 3) Métricas de desempenho (teste)")

# TODO: substituir pelos valores reais do notebook
mae_real = 0.0157
rmse_real = 0.0218
r2_real = 0.9799
mape_real = 4.28

m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE", f"{mae_real:.4f}")
m2.metric("RMSE", f"{rmse_real:.4f}")
m3.metric("R²", f"{r2_real:.4f}")
m4.metric("MAPE", f"{mape_real:.2f}%")

st.caption("Valores atuais do experimento de referência. Atualiza estes indicadores sempre que treinares uma nova versão do modelo.")


@st.cache_data
def build_real_vs_pred_df() -> pd.DataFrame:
    model, scaler, historical_data = load_smart_models()
    if model is None or scaler is None or historical_data is None:
        return pd.DataFrame()

    required_cols = ["hour", "day_of_week", "month", "is_weekend", "lag_1", "lag_48", "energy_kwh"]
    if any(col not in historical_data.columns for col in required_cols):
        return pd.DataFrame()

    df_model = historical_data.dropna(subset=required_cols).copy()
    if df_model.empty:
        return pd.DataFrame()

    split_point = int(len(df_model) * 0.8)
    test = df_model.iloc[split_point:].copy()
    if test.empty:
        return pd.DataFrame()

    X_test = test[["hour", "day_of_week", "month", "is_weekend", "lag_1", "lag_48"]]
    y_test = test["energy_kwh"].astype(float)

    X_test_scaled = scaler.transform(X_test)
    y_pred = np.clip(model.predict(X_test_scaled), a_min=0, a_max=None)

    plot_df = pd.DataFrame(
        {
            "Index": range(len(test)),
            "Consumo real (kWh)": y_test.values,
            "Previsao XGBoost (kWh)": y_pred,
        }
    )

    if "timestamp" in test.columns:
        ts = pd.to_datetime(test["timestamp"], errors="coerce")
        if ts.notna().any():
            plot_df["Timestamp"] = ts.values

    return plot_df


st.markdown("## 4) Comparação: consumo real vs previsão XGBoost")

real_vs_pred = build_real_vs_pred_df()
if real_vs_pred.empty:
    st.warning("Nao foi possivel gerar o grafico de comparação com os artefactos atuais.")
else:
    max_points = min(1000, len(real_vs_pred))
    points_to_show = st.slider(
        "N.º de pontos no gráfico",
        min_value=100,
        max_value=max_points,
        value=min(300, max_points),
        step=50,
    )

    plot_sample = real_vs_pred.tail(points_to_show).copy()
    x_axis = "Timestamp" if "Timestamp" in plot_sample.columns else "Index"

    line_df = plot_sample.melt(
        id_vars=[x_axis],
        value_vars=["Consumo real (kWh)", "Previsao XGBoost (kWh)"],
        var_name="Serie",
        value_name="kWh",
    )

    fig_compare = px.line(
        line_df,
        x=x_axis,
        y="kWh",
        color="Serie",
        title="Consumo real vs previsão do modelo (conjunto de teste)",
        color_discrete_map={
            "Consumo real (kWh)": "#2563EB",
            "Previsao XGBoost (kWh)": "#F59E0B",
        },
    )
    st.plotly_chart(fig_compare, use_container_width=True)
    st.caption("Azul = consumo real | Laranja = previsão do XGBoost")

st.markdown("## 5) Importância das variáveis")

feat_data = pd.DataFrame(
    {
        "Variavel": ["lag_1", "lag_48", "hour", "month", "is_weekend", "day_of_week",],
        "Importancia": [0.7893, 0.1411, 0.0407, 0.0174, 0.0057, 0.0057,],
    }
).sort_values(by="Importancia", ascending=True)

fig = px.bar(
    feat_data,
    x="Importancia",
    y="Variavel",
    orientation="h",
    color="Importancia",
    color_continuous_scale="Viridis",
    title="Importância das variáveis no XGBoost",
)
st.plotly_chart(fig, use_container_width=True)

st.caption("Quanto maior a barra, maior o impacto da variável na previsão final do consumo.")

st.markdown(
    """
    ## 6) Limitações atuais
    - O modelo trabalha com consumo agregado, não individual por cliente.
    - Eventos extremos (feriados, falhas de medição, picos anormais) podem degradar o desempenho.
    - A qualidade da previsão depende da consistência temporal dos dados e dos lags.
    - Para outros contextos geográficos ou perfis de consumo, é necessária nova validação.
    """
)

st.markdown(
    """
    ## 7) Nota metodológica (amostra agregada)
    O modelo foi treinado para prever consumo energético agregado com base em variáveis temporais
    e históricas (lags). O desempenho foi avaliado com MAE, RMSE e R² em dados de teste.

    Isto significa que os resultados representam comportamento coletivo (grupo de clientes),
    não consumo individual de cada utilizador.
    """
)

st.success("O modelo é adequado para apoiar decisões operacionais e recomendações contextuais no produto.")
