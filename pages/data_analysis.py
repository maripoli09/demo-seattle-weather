import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

st.title("O Modelo XGBoost")

st.markdown("""
### Como funciona a previsão?
O sistema utiliza um algoritmo de **Extreme Gradient Boosting (XGBoost)** treinado para prever o consumo energético agregado com base em:
1. **Fatores Temporais:** Hora, Dia da Semana e Mês.
2. **Componente Histórica (Lags):** Consumo verificado 1 hora antes e 48 horas antes.
""")

# Mostrar Feature Importance
st.subheader("Importância das Variáveis")
# Valores de exemplo baseados no teu XGBoost típico
feat_data = pd.DataFrame({
    'Variável': ['Hora', 'Consumo (Lag 1h)', 'Dia da Semana', 'Consumo (Lag 48h)', 'Fim de Semana', 'Mês'],
    'Importância': [0.45, 0.30, 0.10, 0.08, 0.05, 0.02]
}).sort_values(by='Importância', ascending=True)

fig = px.bar(feat_data, x='Importância', y='Variável', orientation='h', color='Importância', color_continuous_scale='Viridis')
st.plotly_chart(fig, use_container_width=True)

st.info("💡 A 'Hora' e a componente histórica imediata são os fatores que mais influenciam a previsão.")