import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
from tariffs import electricity_price
from weather import obtain_local_weather
from utils import LANGUAGES, estimate_solar_production, load_smart_models, make_prediction, generate_recommendation

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
    t = LANGUAGES["PT"]  # Default to PT initially
    language = st.selectbox(t["language_label"], ["PT", "EN"])
    t = LANGUAGES[language]
    st.divider()
    st.subheader(t["location_label"])
    city = st.text_input(t["city_label"], value="Lisboa")
    api_key = st.text_input("OpenWeather API Key", type="password", value="da106eb8a1c706a86299c4ffc3aebba3")
    st.divider()
    st.subheader(t["tariff_label"])
    cycle = st.selectbox(t["cycle_label"], t["cycle_options"])
    price_type = st.selectbox(t["price_label"], t["price_options"])
    st.subheader(t["panel_number_label"])
    num_solar_panels = st.number_input(t["panel_number_label"], min_value=0, value=0)
    st.subheader(t["panel_power_label"])
    panel_wattage = st.number_input(t["panel_power_label"], min_value=0, value=400)
    installed_power_kw = (num_solar_panels * panel_wattage) / 1000.0
    st.caption(f"{t['installed_power_label']}: {installed_power_kw:.2f} kW")


# ── Dados em tempo real ───────────────────────
now = datetime.now()
hour = now.hour
weekday = now.weekday()   # → passado como day_of_week para make_prediction
month = now.month

weather = weather_data(city)
price = electricity_price(hour, weekday, cycle, price_type)

predicted_consumption = make_prediction(model, scaler, historical_data, hour, weekday, month)
cloud_coverage = weather['clouds'] if weather else 0
predicted_production = estimate_solar_production(num_solar_panels, panel_wattage, cloud_coverage)
energy_balance = predicted_production - predicted_consumption

user_name = st.text_input("User Name", value="User")

if predicted_production > predicted_consumption:
    status_line = "A produção solar está a cobrir parte do consumo."
    main_title = "Tens energia solar disponível"
    main_text = "Bom momento para carregar VE ou usar eletrodomésticos."
elif price <= 0.15:
    status_line = "Preço em vazio neste momento."
    main_title = "Bom momento para usar eletrodomésticos"
    main_text = "Preço baixo: aproveita para cargas flexíveis."
elif price > 0.22 and predicted_production < predicted_consumption:
    status_line = "Estás a consumir mais energia do que estás a produzir."
    main_title = "Evita consumos elevados neste momento"
    main_text = "Estás num período tarifário caro. Adia cargas não urgentes."
elif cloud_coverage > 70:
    status_line = "Produção solar limitada pelas nuvens."
    main_title = "Produção solar limitada hoje"
    main_text = "A nebulosidade atual reduz a energia dos painéis."
else:
    status_line = "Este é um bom momento para utilizar eletrodomésticos."
    main_title = "Estado energético estável"
    main_text = "Mantém os consumos mais intensos para horas económicas."


st.info(f"{main_title}\n\n{main_text}")
st.caption(f"Eis o estado energético da tua casa em {city}, às {now.strftime('%H:%M')}.")
st.write(status_line)

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

if energy_balance > 0.1:
    balance_text = "Excedente solar"
else:
    balance_text = "Consumo quase totalmente coberto"

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Consumo previsto", f"{predicted_consumption:.2f} kWh")
    st.caption("Estimativa da IA para esta hora")

with c2:
    st.metric("Produção solar", f"{predicted_production:.2f} kWh")
    st.caption("Com base em painéis e no tempo atual")

with c3:
    st.metric("Balanço atual", f"{energy_balance:.2f} kWh")
    st.caption(balance_text)

meteo_line = (
    f"{weather['temperature']:.1f} °C · "
    f"{weather.get('weather_description', 'sem descrição')} · "
    f"Vento {weather['wind_speed'] * 3.6:.1f} km/h · "
    f"Nebulosidade {cloud_coverage}% · "
    f"Preço {price:.3f} €/kWh"
    if weather else
    f"Meteorologia indisponível · Preço {price:.3f} €/kWh"
)
st.caption(meteo_line)

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

prices = [electricity_price(h, weekday, cycle, price_type) for h in range(24)]
df_tariff = pd.DataFrame({"Hour": list(range(24)), "Price (€/kWh)": prices})

fig = px.line(df_tariff, x="Hour", y="Price (€/kWh)", markers=True,
              title=t["chart_title"], color_discrete_sequence=["#f7a600"])
fig.add_vline(x=hour, line_dash="dash", line_color="red",
              annotation_text=t["now_label"])
st.plotly_chart(fig, use_container_width=True)


st.divider()
st.subheader("Balanço energético nas próximas 24 horas")

rows = []
for i in range(24):
    h = (hour + i) % 24
    d = (weekday + ((hour + i) // 24)) % 7

    consumo_h = make_prediction(model, scaler, historical_data, h, d, month)
    producao_h = estimate_solar_production(num_solar_panels, panel_wattage, cloud_coverage, h)
    balanco_h = producao_h - consumo_h

    rows.append(
        {
            "Hour": h,
            "Consumo_previsto": consumo_h,
            "Producao_solar": producao_h,
            "Balanco": balanco_h,
        }
    )

df_energy = pd.DataFrame(rows)

fig_energy = px.line(
    df_energy,
    x="Hour",
    y=["Consumo_previsto", "Producao_solar"],
    markers=True,
    title="Consumo vs Produção (24h)",
)
fig_energy.add_vline(
    x=hour,
    line_dash="dash",
    line_color="red",
    annotation_text="Agora",
)

st.plotly_chart(fig_energy, use_container_width=True)


with st.expander("Comparar ciclos tarifários"):
    rows_compare = []
    for cycle_name in ["Simple", "Two-cycle", "Three-cycle"]:
        for h in range(24):
            rows_compare.append(
                {"Hour": h, 
                 "Price": electricity_price(h, weekday, cycle_name, price_type), 
                 "Cycle": cycle_name
                 }
            )
    df_compare = pd.DataFrame(rows_compare)
    fig_compare = px.line(
        df_compare, 
        x="Hour", 
        y="Price", 
        color="Cycle", 
        markers=True,
        title="Comparação de ciclos tarifários"
    )
    fig_compare.add_vline(x=hour, line_dash="dash", line_color="red", annotation_text="Agora")
    st.plotly_chart(fig_compare, use_container_width=True)

st.subheader(f"Meteorologia em {city}")
if weather:
    st.write(
        f"{weather['temperature']:.0f} °C · "
        f"{weather.get('weather_description', 'sem descrição')} · "
        f"Vento {weather['wind_speed'] * 3.6:.1f} km/h · "
        f"Nebulosidade {cloud_coverage}%"
    )
else:
    st.write("Dados meteorológicos indisponíveis.")

st.caption("Smart Energy Advisor © 2026 | Powered by XGBoost + OpenWeatherMap")