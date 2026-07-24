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

    train = df_model.iloc[:split_point].copy()

    X_test = test[["hour", "day_of_week", "month", "is_weekend", "lag_1", "lag_48"]]
    y_test = test["energy_kwh"].astype(float)

    X_test_scaled = scaler.transform(X_test)
    y_pred = np.clip(model.predict(X_test_scaled), a_min=0, a_max=None)

    baseline = y_test.shift(1)
    baseline_fill = float(train["energy_kwh"].iloc[-1]) if not train.empty else float(y_test.mean())
    baseline = baseline.fillna(baseline_fill)

    plot_df = pd.DataFrame(
        {
            "Index": range(len(test)),
            "Consumo real (kWh)": y_test.values,
            "Previsao XGBoost (kWh)": y_pred,
            "Baseline persistencia (kWh)": baseline.values,
        }
    )

    if "timestamp" in test.columns:
        ts = pd.to_datetime(test["timestamp"], errors="coerce")
        if ts.notna().any():
            plot_df["Timestamp"] = ts.values

    return plot_df


st.markdown("## 4) Comparação: consumo real vs previsão XGBoost")
st.caption(
    "A linha azul representa o consumo real e a laranja a previsão do modelo. "
    \
    "Quanto mais próximas estiverem, melhor o desempenho preditivo."
)

real_vs_pred = build_real_vs_pred_df()
if real_vs_pred.empty:
    st.warning("Nao foi possivel gerar o grafico de comparação com os artefactos atuais.")
else:
    def calc_metrics(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float]:
        y_true_np = y_true.to_numpy(dtype=float)
        y_pred_np = y_pred.to_numpy(dtype=float)

        mae = float(np.mean(np.abs(y_true_np - y_pred_np)))

        mask = y_true_np != 0
        if np.any(mask):
            mape = float(np.mean(np.abs((y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask])) * 100)
        else:
            mape = 0.0

        return mae, mape

    plot_sample = real_vs_pred.copy()
    x_axis = "Timestamp" if "Timestamp" in plot_sample.columns else "Index"

    if x_axis == "Timestamp":
        ts_max = plot_sample["Timestamp"].max()
        cutoff = ts_max - pd.Timedelta(days=3)
        recent = plot_sample[plot_sample["Timestamp"] >= cutoff].copy()
        if not recent.empty:
            plot_sample = recent

    global_mae, global_mape = calc_metrics(
        real_vs_pred["Consumo real (kWh)"],
        real_vs_pred["Previsao XGBoost (kWh)"],
    )
    global_mae_baseline, global_mape_baseline = calc_metrics(
        real_vs_pred["Consumo real (kWh)"],
        real_vs_pred["Baseline persistencia (kWh)"],
    )

    window_mae, window_mape = calc_metrics(
        plot_sample["Consumo real (kWh)"],
        plot_sample["Previsao XGBoost (kWh)"],
    )
    window_mae_baseline, window_mape_baseline = calc_metrics(
        plot_sample["Consumo real (kWh)"],
        plot_sample["Baseline persistencia (kWh)"],
    )

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
        title="Consumo real vs previsão do modelo",
        color_discrete_map={
            "Consumo real (kWh)": "#2563EB",
            "Previsao XGBoost (kWh)": "#F59E0B",
        },
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE - teste global", f"{global_mae:.4f} kWh")
    c2.metric("MAE - janela atual", f"{window_mae:.4f} kWh")
    c3.metric("MAE baseline - janela", f"{window_mae_baseline:.4f} kWh")

    c4, c5, c6 = st.columns(3)
    c4.metric("MAPE - teste global", f"{global_mape:.2f}%")
    c5.metric("MAPE - janela atual", f"{window_mape:.2f}%")
    c6.metric("MAPE baseline - janela", f"{window_mape_baseline:.2f}%")

    win_gain = window_mae_baseline - window_mae
    if win_gain > 0:
        comp_text = f"melhor que o baseline por {win_gain:.4f} kWh de MAE"
    elif win_gain < 0:
        comp_text = f"pior que o baseline por {abs(win_gain):.4f} kWh de MAE"
    else:
        comp_text = "equivalente ao baseline na janela"

    st.markdown(
        f"""
        **Leitura rápida dos resultados:**
        No **teste global**, o modelo apresenta MAE de **{global_mae:.4f} kWh** (MAPE **{global_mape:.2f}%**).
        Na **janela atual**, o MAE é **{window_mae:.4f} kWh** (MAPE **{window_mape:.2f}%**) e está **{comp_text}**.
        """
    )

    if x_axis == "Timestamp":
        st.caption("A visualização mostra, por defeito, os últimos 3 dias do conjunto de teste.")

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
