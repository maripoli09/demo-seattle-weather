import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
from tariffs import electricity_price
from weather import obtain_local_weather
from utils import LANGUAGES, load_smart_models, make_prediction, generate_recommendation

weather_data = obtain_local_weather 

# Configurations of the page
st.set_page_config(page_title="Smart Energy Advisor", layout="wide")

@st.cache_resource
def load_all():
    return load_smart_models()

model, scaler, historical_data = load_all()

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.title("Definições")
    language = st.selectbox("Língua / Language", ["PT", "EN"])
    t = LANGUAGES[language]
    st.divider()
    city = st.text_input("Cidade / City", value="Lisboa")
    api_key = st.text_input("OpenWeather API Key", type="password", value="da106eb8a1c706a86299c4ffc3aebba3")
    st.divider()
    st.subheader("Tarifa")
    cycle = st.selectbox("Ciclo / Cycle", ["Simple", "Two-cycle", "Three-cycle"])
    model_type = st.selectbox("Modelo / Model", ["Fixed", "Variable"])
    st.subheader("Número de Painéis Solares / Number of Solar Panels")
    num_solar_panels = st.number_input("Número de Painéis Solares / Number of Solar Panels", min_value=0, value=0)
    st.subheader("Potência de cada Painel / Power of each Panel (W)")
    panel_wattage = st.number_input("Potência de cada Painel / Power of each Panel (W)", min_value=0, value=450)
    st.divider()

# ── Dados em tempo real ───────────────────────
now = datetime.now()
hour = now.hour
weekday = now.weekday()   # → passado como day_of_week para make_prediction
month = now.month

weather = weather_data(city)
price = electricity_price(hour, weekday, cycle, model_type)


# Title 
st.title(f"Smart Energy Advisor - {city}")
st.caption(f"{t['footer']}: {now.strftime('%Y-%m-%d %H:%M:%S')}")

# Dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        t["temp"], 
        f"{weather['temperature']}ºC" if weather else "N/A", 
        delta=f"Humidade: {weather['humidity']}%" if weather else None
    )

with col2:
    st.metric(t["price"], f"{price:.3f} €/kWh")

with col3:
    st.metric(t["clouds_label"], f"{weather['clouds']}%" if weather else "N/A")

with col4:
    wind_kmh = weather["wind_speed"] * 3.6 if weather else None

    st.metric(t["wind_label"], f"{wind_kmh:.1f} km/h" if wind_kmh is not None else "N/A",)

st.divider()

# Prevision
st.subheader(f"{t['forecast_section']}")

predicted_consumption = make_prediction(model, scaler, historical_data, hour, weekday, month)
predicted_production = 0.0

col_pred1, col_pred2 = st.columns(2)
with col_pred1:
    st.metric(f"{t['prediction_text']}", f"{predicted_consumption:.3f} kWh")
with col_pred2:
    st.metric(t["solar_label"], f"{predicted_production:.3f} kWh")

st.divider()

# Recommendations
st.subheader(t["rec_title"])

cloud_couverage = weather['clouds'] if weather else 0
advices = generate_recommendation(predicted_consumption, predicted_production, price, cloud_couverage, t)

for advice in advices:
    st.info(f"{advice}")

st.divider()


# Graphs
st.subheader(t["hourly_chart"])

prices = [electricity_price(h, weekday, cycle, model_type) for h in range(24)]
df_tariff = pd.DataFrame({"Hour": list(range(24)), "Price (€/kWh)": prices})

fig = px.line(df_tariff, x="Hour", y="Price (€/kWh)", markers=True,
              title=t["chart_title"], color_discrete_sequence=["#f7a600"])
fig.add_vline(x=hour, line_dash="dash", line_color="red",
              annotation_text=t["now_label"])
st.plotly_chart(fig, use_container_width=True)

st.caption("Smart Energy Advisor © 2026 | Powered by XGBoost + OpenWeatherMap")