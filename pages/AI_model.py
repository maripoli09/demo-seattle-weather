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
        {"Variavel": "lag_1", "Descricao": "Consumo observado 1 hora antes"},
        {"Variavel": "lag_48", "Descricao": "Consumo observado 48 horas antes"},
    ]
)
st.dataframe(features_df, use_container_width=True)

st.markdown("## 3) Metricas reais do notebook (teste)")

# TODO: substituir pelos valores reais do notebook
mae_real = 0.0000
rmse_real = 0.0000
r2_real = 0.0000

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae_real:.4f}")
m2.metric("RMSE", f"{rmse_real:.4f}")
m3.metric("R²", f"{r2_real:.4f}")

st.info(
    "Substitui os placeholders (0.0000) pelos resultados reais obtidos no notebook."
)

st.markdown("## 4) Importancia das variaveis (modelo real)")

# TODO: substituir pelos valores reais extraidos do modelo treinado
feat_data = pd.DataFrame(
    {
        "Variavel": ["hour", "lag_1", "day_of_week", "lag_48", "is_weekend", "month"],
        "Importancia": [0.45, 0.30, 0.10, 0.08, 0.05, 0.02],
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
    Neste trabalho, a modelacao foi feita sobre dados agregados de 300 clientes do dataset Ausgrid.
    Esta abordagem melhora estabilidade estatistica e reduz ruido individual, mas perde granularidade
    de comportamento por consumidor. Assim, os resultados devem ser interpretados ao nivel de perfil
    energetico agregado e nao como previsao personalizada por cliente.
    """
)

with st.expander("Texto curto para usar no relatorio"):
    st.markdown(
        """
        O modelo XGBoost foi utilizado para prever consumo energetico agregado com base em variaveis
        temporais e historicas (lags). O desempenho foi avaliado por MAE, RMSE e R² em dados de teste.
        A analise de importancia das variaveis permitiu identificar os principais fatores explicativos.
        Como limitacao, o modelo opera sobre agregacao de 300 clientes, sendo os resultados representativos
        de comportamento coletivo e nao individual.
        """
    )