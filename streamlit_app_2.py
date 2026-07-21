import streamlit as st
import importlib
import pandas as pd
from datetime import datetime
import plotly.express as px
from supabase import create_client, Client
from tariffs import electricity_price
from weather import obtain_local_weather, future_weather
from utils import LANGUAGES, estimate_solar_production, load_smart_models, make_prediction, generate_recommendation

# Recover from previous runs where st.write may have been accidentally overwritten.
if not callable(getattr(st, "write", None)):
    st = importlib.reload(st)

@st.cache_resource
def load_all():
    return load_smart_models()

model, scaler, historical_data = load_all()

@st.cache_resource
def get_supabase_client() -> Client | None:
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)




def save_simulation(payload: dict) -> tuple[bool, str]:
    try:
        supabase = get_supabase_client()
        if supabase is None:
            return False, "Supabase não configurado."
        supabase.table("simulations").insert(payload).execute()
        return True, "Simulação guardada com sucesso."
    except Exception as e:
        return False, f"Erro ao guardar: {e}"

# Session state defaults required by popup and main page.
STATE_DEFAULTS = {
    "abrir_config": True,
    "city": "Lisboa",
    "cycle": "Simples",
    "price_type": "Preço fixo",
    "num_solar_panels": 0,
    "panel_wattage": 400,
    "language": "PT",
}
for _key, _value in STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value


@st.dialog("Configurar habitação")
def popup_configuracao():
    st.subheader("Dados da habitação")
    city_options = ["Lisboa", "Porto", "Coimbra", "Faro", "Funchal"]
    cycle_options = ["Simples", "Bi-horária", "Tri-horária"]
    price_options = ["Preço fixo", "Preço variável"]

    # Mapear os inputs diretamente para o session_state para não perder as escolhas
    st.session_state.city = st.selectbox(
        "Cidade",
        city_options,
        index=city_options.index(st.session_state.city) if st.session_state.city in city_options else 0,
    )

    st.session_state.cycle = st.selectbox(
        "Ciclo tarifário",
        cycle_options,
        index=cycle_options.index(st.session_state.cycle) if st.session_state.cycle in cycle_options else 0,
    )

    st.session_state.price_type = st.selectbox(
        "Tipo de preço",
        price_options,
        index=price_options.index(st.session_state.price_type) if st.session_state.price_type in price_options else 0,
    )

    st.session_state.num_solar_panels = st.number_input(
        "Número de painéis solares",
        min_value=0,
        value=st.session_state.num_solar_panels,
        step=1,
    )


    st.session_state.panel_wattage = st.number_input(
        "Potência de cada painel (W)",
        min_value=0,
        value=st.session_state.panel_wattage,
        step=50,
    )

    installed_power_kw = (st.session_state.num_solar_panels * st.session_state.panel_wattage) / 1000
    st.caption(f"Potência instalada: {installed_power_kw:.2f} kW")
    
    # Botão para salvar e fechar a janela modal
    if st.button("Gravar Configurações", use_container_width=True):
        st.rerun()

# 4. Executar o Pop-up se a condição for verdadeira
if st.session_state.abrir_config:
    st.session_state.abrir_config = False  # Evita que abra em loop nas próximas interações
    popup_configuracao()

city = st.session_state.city
cycle = st.session_state.cycle
price_type = st.session_state.price_type
num_solar_panels = st.session_state.num_solar_panels
panel_wattage = st.session_state.panel_wattage

if st.button("Alterar Configurações"):
    st.session_state.abrir_config = True
    st.rerun()

weather_data = obtain_local_weather

# Configurations of the page
st.set_page_config(page_title="Smart Energy Advisor", layout="wide")


# ── Sidebar ──────────────────────────────────
with st.sidebar:
    t = LANGUAGES["PT"]  # Default to PT initially
    st.title(t["definitions"])
    language = st.selectbox(
        t["language_label"],
        ["PT", "EN"],
        index=["PT", "EN"].index(st.session_state.language) if st.session_state.language in ["PT", "EN"] else 0,
    )
    st.session_state.language = language
    t = LANGUAGES[language]
    st.divider()



# ── Dados em tempo real ───────────────────────
now = datetime.now()
hour = now.hour
weekday = now.weekday()   # → passado como day_of_week para make_prediction
month = now.month
weather = weather_data(city)
price = electricity_price(hour, weekday, cycle, price_type)

predicted_consumption = make_prediction(model, scaler, historical_data, hour, weekday, month)
cloud_coverage = weather['clouds'] if weather else 0
predicted_production = estimate_solar_production(
    num_solar_panels,
    panel_wattage,
    cloud_coverage,
    hour=hour,
    temp_c=weather.get("temperature") if weather else None,
)
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
st.title("Bem-vindo ao Smart Energy Advisor")
st.markdown("""
### Escolha uma opção no menu lateral:
1. **Dashboard Principal:** Previsão em tempo real e simulação solar.
2. **Análise de Dados:** Exploração do dataset Ausgrid (300 clientes).
3. **Modelo IA:** Detalhes técnicos e métricas do XGBoost.
4. **Histórico:** Registos guardados no Supabase.
""")

st.info("Podes configurar a tua localização e tarifário na barra lateral de qualquer página.")


tab_current, tab_prevision, tab_simulator = st.tabs(
    ["Informações atuais", "Previsão do dia", "Simulador e poupança"]
)

with tab_current:
   
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

weather_forecast = future_weather(city)

forecast_rows = []

if weather_forecast:
    for item in weather_forecast:
        forecast_time = datetime.fromtimestamp(item["dt"])

        # Mantém apenas as previsões até ao fim do dia atual.
        if forecast_time.date() != now.date():
            continue

        forecast_hour = forecast_time.hour
        forecast_clouds = item.get("clouds", {}).get("all", 0)
        forecast_temp = item.get("main", {}).get("temp")
        forecast_description = item.get("weather", [{}])[0].get(
            "description",
            "Sem descrição",
        )

        consumption = make_prediction(
            model,
            scaler,
            historical_data,
            forecast_hour,
            forecast_time.weekday(),
            forecast_time.month,
        )

        production = estimate_solar_production(
            num_solar_panels,
            panel_wattage,
            forecast_clouds,
            hour=forecast_hour,
            month=forecast_time.month,
            temp_c=forecast_temp,
            )
        
        forecast_price = electricity_price(
            forecast_hour,
            forecast_time.weekday(),
            cycle,
            price_type,
        )

        balance = production - consumption
        grid_energy = max(consumption - production, 0)
        estimated_cost = grid_energy * forecast_price

        forecast_rows.append(
            {
                "Hora": forecast_time.strftime("%H:%M"),
                "Condição": forecast_description.capitalize(),
                "Temperatura (°C)": forecast_temp,
                "Nebulosidade (%)": forecast_clouds,
                "Preço (€/kWh)": forecast_price,
                "Consumo previsto (kWh)": consumption,
                "Produção solar (kWh)": production,
                "Balanço (kWh)": balance,
                "Custo estimado (€)": estimated_cost,
            }
        )

df_forecast = pd.DataFrame(forecast_rows)

with tab_prevision:
    st.subheader("Plano energético previsto para hoje")
    st.caption(
        "A previsão meteorológica é obtida em intervalos de 3 horas. "
        "O consumo previsto corresponde ao perfil agregado estimado pelo modelo XGBoost."
    )

    if df_forecast.empty:
        st.warning("Não foi possível obter previsões para hoje.")
    else:
        melhor_periodo = df_forecast.loc[
            df_forecast["Custo estimado (€)"].idxmin()
        ]

        st.success(
            f"**Melhor hora para consumos flexíveis: "
            f"{melhor_periodo['Hora']}**  \n"
            f"Custo estimado de energia comprada à rede: "
            f"{melhor_periodo['Custo estimado (€)']:.3f} €."
        )

        st.dataframe(
            df_forecast,
            use_container_width=True,
            hide_index=True,
        )    

        fig_forecast = px.line(
    df_forecast,
    x="Hora",
    y=[
        "Consumo previsto (kWh)",
        "Produção solar (kWh)",
    ],
    markers=True,
    title="Consumo previsto vs. produção solar ao longo do dia",
)

st.plotly_chart(fig_forecast, use_container_width=True)


  
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
    balance_text = "Excedente solar disponível"
elif energy_balance >= -0.1:
    balance_text = "Consumo quase totalmente coberto"
else:
    balance_text = "Necessário importar energia da rede"

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

forecast_clouds_by_hour = {}
forecast_temp_by_hour = {}

if weather_forecast:
    for item in weather_forecast:
        ft = datetime.fromtimestamp(item["dt"])
        if ft.date() != now.date():
            continue

        fh = ft.hour
        forecast_clouds_by_hour[fh] = item.get("clouds", {}).get("all", cloud_coverage)
        forecast_temp_by_hour[fh] = item.get("main", {}).get("temp", weather.get("temperature") if weather else None)

rows = []
for i in range(24):
    h = (hour + i) % 24
    d = (weekday + ((hour + i) // 24)) % 7
    consumo_h = make_prediction(model, scaler, historical_data, h, d, month)

    clouds_h = forecast_clouds_by_hour.get(h, cloud_coverage)
    temp_h = forecast_temp_by_hour.get(h, weather.get("temperature") if weather else None)

    producao_h = estimate_solar_production(
        num_solar_panels,
        panel_wattage,
        clouds_h,
        hour=h,
        temp_c=temp_h,
    )
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

df_energy["Preco"] = [
    electricity_price(
        int(hora),
        weekday,
        cycle,
        price_type,
    )
    for hora in df_energy["Hour"]
]

df_energy["Custo_liquido"] = (
    (df_energy["Consumo_previsto"] - df_energy["Producao_solar"])
    .clip(lower=0)
    * df_energy["Preco"]
)
melhor_hora = df_energy.loc[df_energy["Custo_liquido"].idxmin()]

st.success(
    f"### Melhor hora para consumos flexíveis: {int(melhor_hora['Hour']):02d}:00\n"
    f"Nessa hora, o custo estimado da energia comprada à rede é "
    f"{melhor_hora['Custo_liquido']:.3f} €."
)

st.divider()
st.subheader("Simulador de custo e poupanca (24h)")

# custo sem solar: todo o consumo comprado a rede
custo_sem_solar = 0.0
# custo com solar: compra apenas deficit
custo_com_solar = 0.0

for i in range(24):
    h = int(df_energy.loc[i, "Hour"])
    preco_h = electricity_price(h, weekday, cycle, price_type)
    consumo_h = float(df_energy.loc[i, "Consumo_previsto"])
    producao_h = float(df_energy.loc[i, "Producao_solar"])

    custo_sem_solar += consumo_h * preco_h
    deficit = max(consumo_h - producao_h, 0.0)
    custo_com_solar += deficit * preco_h

poupanca = custo_sem_solar - custo_com_solar

c_a, c_b, c_c = st.columns(3)
with c_a:
    st.metric("Custo sem solar (24h)", f"{custo_sem_solar:.2f} €")
with c_b:
    st.metric("Custo com solar (24h)", f"{custo_com_solar:.2f} €")
with c_c:
    st.metric("Poupanca estimada (24h)", f"{poupanca:.2f} €")


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


simulation_payload = {
    "city": city,
    "cycle": cycle,
    "price_type": price_type,
    "num_solar_panels": int(num_solar_panels),
    "panel_wattage": int(panel_wattage),
    "predicted_consumption": float(predicted_consumption),
    "predicted_production": float(predicted_production),
    "energy_balance": float(energy_balance),
    "price_now": float(price),
    "estimated_cost_without_solar": float(custo_sem_solar) if "custo_sem_solar" in locals() else None,
    "estimated_cost_with_solar": float(custo_com_solar) if "custo_com_solar" in locals() else None,
    "estimated_savings": float(poupanca) if "poupanca" in locals() else None,
    "model_version": "xgboost_v1"
}

st.divider()
if st.button("Guardar simulação no histórico"):
    ok, msg = save_simulation(simulation_payload)
    if ok:
        st.success(msg)
    else:
        st.warning(msg)

       

st.caption("Smart Energy Advisor © 2026 | Powered by XGBoost + OpenWeatherMap")