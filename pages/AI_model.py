import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Modelo IA", layout="wide")
st.title("Modelo IA + Explicabilidade")
st.caption("Previsao de consumo energetico agregado com XGBoost")

st.markdown(
    """
    ## 1) O que e o XGBoost e porque foi escolhido
    O XGBoost (Extreme Gradient Boosting) e um algoritmo de ensemble baseado em arvores de decisao
    que combina varias arvores fracas para gerar um modelo mais robusto.
    
    Foi escolhido neste projeto por:
    - modelar relacoes nao lineares entre variaveis temporais e consumo
    - lidar bem com dados tabulares
    - oferecer boa capacidade preditiva em series energeticas agregadas
    - permitir interpretar a importancia relativa das variaveis
    """
)

st.markdown("## 2) Variaveis usadas no modelo")
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

st.markdown("## 3) Metricas reais do notebook (teste)")

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

st.info(
    "Substitui os placeholders (0.0000) pelos resultados reais obtidos no notebook."
)

st.markdown("## 4) Importancia das variaveis (modelo real)")

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
    title="Importancia das variaveis no XGBoost",
)
st.plotly_chart(fig, use_container_width=True)

st.warning(
    "Troca estes valores pelos reais do teu modelo para ficar metodologicamente correto."
)

st.markdown(
    """
    ## 5) Limitacoes do modelo
    - O modelo trabalha com consumo agregado, nao individual por cliente.
    - Eventos extremos (feriados, falhas de medicao, picos anormais) podem degradar o desempenho.
    - O modelo depende da qualidade dos lags e da consistencia temporal dos dados.
    - Generalizacao para outros contextos depende de novas validacoes.
    """
)

st.markdown(
    """
    ## 6) Nota metodologica sobre agregacao (300 clientes)
    O modelo XGBoost foi utilizado para prever consumo energetico agregado com base em variaveis
    temporais e historicas (lags). O desempenho foi avaliado por MAE, RMSE e R² em dados de teste.
    A analise de importancia das variaveis permitiu identificar os principais fatores explicativos.
    Como limitacao, o modelo opera sobre agregacao de 300 clientes, sendo os resultados representativos
    de comportamento coletivo e nao individual.
    """
)
