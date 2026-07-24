import importlib
import re
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from supabase import create_client
except ModuleNotFoundError:
    create_client = None

from tariffs import electricity_price
from utils import (
    estimate_solar_production,
    generate_recommendation,
    load_smart_models,
    make_prediction,
)
from weather import future_weather, obtain_local_weather


st.set_page_config(page_title="Smart Energy Advisor", layout="wide")


# Recover from a previous run that may have overwritten st.write in memory.
if not callable(getattr(st, "write", None)):
    st = importlib.reload(st)


CITY_OPTIONS = ["Lisboa", "Porto", "Coimbra", "Faro", "Funchal"]
CYCLE_OPTIONS = ["Simples", "Bi-horária", "Tri-horária"]
PRICE_OPTIONS = ["Preço fixo", "Preço variável"]
STATE_DEFAULTS = {
    "abrir_config": True,
    "city": "Lisboa",
    "cycle": "Simples",
    "price_type": "Preço fixo",
    "num_solar_panels": 0,
    "panel_wattage": 400,
}

PT_TEXTS = {
    "temp": "Temperatura",
    "price": "Preço Atual",
    "hourly_chart": "Horário Tarifário",
    "forecast_section": "Previsão de Consumo (IA)",
    "clouds_label": "Nebulosidade",
    "wind_label": "Vento",
    "rec_title": "Recomendações",
    "footer": "Atualizado em",
    "now_label": "Agora",
    "chart_title": "Evolução do Preço da Eletricidade ao Longo do Dia",
    "rec_low_price": "Preço baixo — Bom momento para utilizar eletricidade.",
    "rec_high_price": "Preço elevado — Considera reduzir o consumo elétrico.",
    "rec_high_solar": "Produção solar elevada — Recomenda-se o uso de energia solar.",
    "rec_partial_solar": "Produção solar ativa mas insuficiente — Tenta reduzir o consumo para maximizar o autoconsumo.",
    "rec_clouds": "Nebulosidade elevada — A produção solar pode ser reduzida.",
    "rec_none": "Sem recomendações específicas neste momento.",
}


@st.cache_resource
def load_all():
    return load_smart_models()


def get_supabase_client(authenticated: bool = False) -> Any | None:
    if create_client is None:
        return None

    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        return None

    supabase = create_client(url, key)

    if authenticated:
        access_token = st.session_state.get("access_token")
        refresh_token = st.session_state.get("refresh_token")

        if not access_token or not refresh_token:
            return None

        supabase.auth.set_session(access_token, refresh_token)

    return supabase


def save_simulation(payload: dict[str, Any]) -> tuple[bool, str]:
    try:
        supabase = get_supabase_client(authenticated=True)
        if supabase is None:
            return False, "Sessao invalida ou Supabase nao configurado."
        supabase.table("simulations").insert(payload).execute()
        return True, "Simulation saved successfully."
    except Exception as e:
        return False, f"Error saving simulation: {e}"


def ensure_profile_exists() -> tuple[bool, str]:
    user = st.session_state.get("user")
    if user is None:
        return False, "Sessao invalida."

    supabase = get_supabase_client(authenticated=True)
    if supabase is None:
        return False, "Sessao invalida ou Supabase nao configurado."

    try:
        profile_payload = {"id": user.id}
        user_name = st.session_state.get("user_name")
        if user_name:
            profile_payload["user_name"] = user_name

        supabase.table("profiles").upsert(profile_payload, on_conflict="id").execute()
        return True, ""
    except Exception as e:
        return False, f"Erro ao garantir perfil: {e}"


def init_session_state() -> None:
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def safe_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


def get_config() -> dict[str, Any]:
    return {
        "city": st.session_state.city,
        "cycle": st.session_state.cycle,
        "price_type": st.session_state.price_type,
        "num_solar_panels": st.session_state.num_solar_panels,
        "panel_wattage": st.session_state.panel_wattage,
    }


def signup_error_message(error: Exception) -> str:
    """Map common Supabase sign-up failures to user-friendly PT messages."""
    raw_error = str(error).lower()

    if "already" in raw_error or "exists" in raw_error or "registered" in raw_error:
        return "Este email já está registado. Tenta iniciar sessão."
    if "password" in raw_error and "6" in raw_error:
        return "A palavra-passe deve ter pelo menos 6 caracteres."
    if "invalid email" in raw_error or "email" in raw_error and "invalid" in raw_error:
        return "O email introduzido não é válido."

    return f"Erro ao criar conta: {error}"


def password_meets_complexity(password: str) -> bool:
    """Require uppercase, number and symbol for signup passwords."""
    has_uppercase = re.search(r"[A-Z]", password) is not None
    has_number = re.search(r"[0-9]", password) is not None
    has_symbol = re.search(r"[^A-Za-z0-9]", password) is not None
    return has_uppercase and has_number and has_symbol


def extract_user_name(user: Any) -> str:
    """Get username from metadata with sensible fallback."""
    metadata = getattr(user, "user_metadata", {}) or {}
    user_name = metadata.get("user_name")
    if user_name:
        return str(user_name)

    email = getattr(user, "email", "") or ""
    if "@" in email:
        return email.split("@", 1)[0]
    return "utilizador"


def resolve_user_name(user: Any) -> str:
    """Resolve username from session, metadata or profile, without exposing full email."""
    session_user_name = (st.session_state.get("user_name") or "").strip()
    if session_user_name:
        return session_user_name

    metadata_user_name = extract_user_name(user)
    if metadata_user_name and metadata_user_name != "utilizador":
        return metadata_user_name

    supabase = get_supabase_client(authenticated=True)
    if supabase is not None and getattr(user, "id", None):
        try:
            result = (
                supabase.table("profiles")
                .select("user_name")
                .eq("id", user.id)
                .limit(1)
                .execute()
            )
            profile_rows = getattr(result, "data", []) or []
            if profile_rows:
                profile_user_name = (profile_rows[0].get("user_name") or "").strip()
                if profile_user_name:
                    return profile_user_name
        except Exception:
            pass

    return "utilizador"

if "user" not in st.session_state:
    st.session_state.user = None

if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = None

if "user_name" not in st.session_state:
    st.session_state.user_name = None

@st.dialog("Iniciar sessão")
def popup_login() -> None:
    st.caption("Inicia sessão para guardares e consultares os teus cenários.")

    supabase = get_supabase_client()
    if supabase is None:
        st.error("Supabase não configurado.")
        return

    email = st.text_input("Email")
    password = st.text_input("Palavra-passe", type="password")
    user_name_input = st.text_input("User name", placeholder="Ex.: joao123")
    feedback = st.empty()

    col_login, col_register = st.columns(2)

    with col_login:
        if st.button("Entrar", use_container_width=True):
            try:
                response = supabase.auth.sign_in_with_password(
                    {"email": email, "password": password}
                )
                st.session_state.user = response.user
                st.session_state.access_token = response.session.access_token
                st.session_state.refresh_token = response.session.refresh_token
                if user_name_input.strip():
                    st.session_state.user_name = user_name_input.strip()
                else:
                    st.session_state.user_name = resolve_user_name(response.user)
                ok_profile, profile_msg = ensure_profile_exists()
                if not ok_profile:
                    feedback.warning(profile_msg)
                feedback.success(f"Bem-vindo, {st.session_state.user_name}!")
                st.rerun()
            except Exception as e:
                feedback.error(f"Erro ao iniciar sessão: {e}")

    with col_register:
        if st.button("Criar conta", use_container_width=True):
            if not email or not password:
                feedback.error("Preenche o email e a palavra-passe para criar conta.")
                return
            if not user_name_input.strip():
                feedback.error("Preenche também o user name para criar conta.")
                return
            if len(password) < 6:
                feedback.error("A palavra-passe deve ter pelo menos 6 caracteres.")
                return
            if not password_meets_complexity(password):
                feedback.error(
                    "A palavra-passe deve incluir pelo menos 1 letra maiúscula, 1 número e 1 símbolo."
                )
                return

            try:
                response = supabase.auth.sign_up(
                    {
                        "email": email,
                        "password": password,
                        "options": {"data": {"user_name": user_name_input.strip()}},
                    }
                )
                if response.session is not None:
                    st.session_state.user = response.user
                    st.session_state.access_token = response.session.access_token
                    st.session_state.refresh_token = response.session.refresh_token
                    st.session_state.user_name = user_name_input.strip()
                    ok_profile, profile_msg = ensure_profile_exists()
                    if not ok_profile:
                        feedback.warning(profile_msg)
                feedback.success(
                    "Conta criada! Verifica o teu email para confirmar o registo."
                )
            except Exception as e:
                feedback.error(signup_error_message(e))

@st.dialog("Configurar habitacao")
def popup_configuracao() -> None:
    st.subheader("Dados da habitacao")

    city_input = st.text_input(
        "Cidade (Portugal)",
        value=st.session_state.city,
        placeholder="Ex.: Braga, Aveiro, Viseu, Setubal",
        help="Podes escrever qualquer cidade portuguesa.",
    )
    if city_input.strip():
        st.session_state.city = city_input.strip()
    st.session_state.cycle = st.selectbox(
        "Ciclo tarifario",
        CYCLE_OPTIONS,
        index=safe_index(CYCLE_OPTIONS, st.session_state.cycle),
    )
    st.session_state.price_type = st.selectbox(
        "Tipo de preco",
        PRICE_OPTIONS,
        index=safe_index(PRICE_OPTIONS, st.session_state.price_type),
    )
    st.session_state.num_solar_panels = st.number_input(
        "Numero de paineis solares",
        min_value=0,
        value=int(st.session_state.num_solar_panels),
        step=1,
    )
    st.session_state.panel_wattage = st.number_input(
        "Potencia de cada painel (W)",
        min_value=0,
        value=int(st.session_state.panel_wattage),
        step=50,
    )

    installed_power_kw = (
        st.session_state.num_solar_panels * st.session_state.panel_wattage
    ) / 1000
    st.caption(f"Potencia instalada: {installed_power_kw:.2f} kW")

    if st.button("Gravar configuracoes", use_container_width=True):
        st.session_state.abrir_config = False
        st.rerun()

# --- Barra de topo ---
col_title, col_buttons = st.columns([3, 1])
dialog_opened = False


with col_buttons:
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        # Mostrar botão de login ou de logout consoante o estado
        if st.session_state.get("user") is None:
            if st.button("Entrar", use_container_width=True):
                popup_login()
                dialog_opened = True
        else:
            user_label = resolve_user_name(st.session_state.user)
            st.session_state.user_name = user_label
            st.caption(f"{user_label}")
            if st.button("Sair", use_container_width=True):
                supabase = get_supabase_client(authenticated=True)
                if supabase is not None:
                    supabase.auth.sign_out()
                st.session_state.user = None
                st.session_state.access_token = None
                st.session_state.refresh_token = None
                st.session_state.user_name = None
                st.rerun()

    with btn_col2:
        if st.button("Configurar", use_container_width=True):
            popup_configuracao()
            dialog_opened = True


def render_header(t: dict[str, Any], city: str, now: datetime) -> None:
    st.title(f"Smart Energy Advisor - {city}")
    st.caption(f"{t['footer']}: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown(
        """
### Escolhe a secao nas tabs abaixo
- Resumo rapido com estado atual
- Previsao detalhada do dia
- Simulador e poupanca
- Configuracoes e historico
"""
    )


def weather_for_city(city: str) -> dict[str, Any] | None:
    try:
        return obtain_local_weather(city)
    except Exception as e:
        st.warning(f"Nao foi possivel obter o clima atual: {e}")
        return None


def build_hourly_weather_lookup(
    weather_forecast: list[dict[str, Any]] | None,
    now: datetime,
    fallback_clouds: float,
    fallback_temp: float | None,
) -> dict[int, dict[str, Any]]:
    if not weather_forecast:
        return {}

    anchors: dict[int, dict[str, Any]] = {}
    for item in weather_forecast:
        forecast_time = datetime.fromtimestamp(item["dt"])
        if forecast_time.date() != now.date():
            continue

        forecast_hour = forecast_time.hour
        anchors[forecast_hour] = {
            "clouds": float(item.get("clouds", {}).get("all", fallback_clouds)),
            "temp": item.get("main", {}).get("temp", fallback_temp),
            "description": item.get("weather", [{}])[0].get("description", "Sem descricao"),
            "weekday": forecast_time.weekday(),
            "month": forecast_time.month,
        }

    if not anchors:
        return {}

    anchor_hours = sorted(anchors.keys())
    start_hour = anchor_hours[0]
    end_hour = anchor_hours[-1]
    hourly_lookup: dict[int, dict[str, Any]] = {}

    for hour in range(start_hour, end_hour + 1):
        if hour in anchors:
            hourly_lookup[hour] = anchors[hour]
            continue

        prev_candidates = [h for h in anchor_hours if h < hour]
        next_candidates = [h for h in anchor_hours if h > hour]
        prev_hour = max(prev_candidates) if prev_candidates else None
        next_hour = min(next_candidates) if next_candidates else None

        if prev_hour is not None and next_hour is not None:
            prev_data = anchors[prev_hour]
            next_data = anchors[next_hour]
            ratio = (hour - prev_hour) / (next_hour - prev_hour)

            prev_temp = prev_data["temp"]
            next_temp = next_data["temp"]
            if prev_temp is not None and next_temp is not None:
                temp = float(prev_temp) + (float(next_temp) - float(prev_temp)) * ratio
            elif prev_temp is not None:
                temp = prev_temp
            else:
                temp = next_temp

            clouds = float(prev_data["clouds"]) + (
                float(next_data["clouds"]) - float(prev_data["clouds"])
            ) * ratio
            description = prev_data["description"] if ratio < 0.5 else next_data["description"]
            weekday = prev_data["weekday"]
            month = prev_data["month"]
        elif prev_hour is not None:
            prev_data = anchors[prev_hour]
            temp = prev_data["temp"]
            clouds = prev_data["clouds"]
            description = prev_data["description"]
            weekday = prev_data["weekday"]
            month = prev_data["month"]
        elif next_hour is not None:
            next_data = anchors[next_hour]
            temp = next_data["temp"]
            clouds = next_data["clouds"]
            description = next_data["description"]
            weekday = next_data["weekday"]
            month = next_data["month"]
        else:
            temp = fallback_temp
            clouds = fallback_clouds
            description = "Sem descricao"
            weekday = now.weekday()
            month = now.month

        hourly_lookup[hour] = {
            "clouds": float(clouds),
            "temp": temp,
            "description": description,
            "weekday": weekday,
            "month": month,
        }

    return hourly_lookup


def build_current_forecast_frame(
    weather_hourly_lookup: dict[int, dict[str, Any]],
    model: Any,
    scaler: Any,
    historical_data: Any,
    cycle: str,
    price_type: str,
    num_solar_panels: int,
    panel_wattage: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not weather_hourly_lookup:
        return pd.DataFrame(rows)

    for forecast_hour in sorted(weather_hourly_lookup.keys()):
        weather_row = weather_hourly_lookup[forecast_hour]
        forecast_clouds = float(weather_row["clouds"])
        forecast_temp = weather_row["temp"]
        forecast_description = weather_row["description"]
        forecast_weekday = int(weather_row["weekday"])
        forecast_month = int(weather_row["month"])

        consumption = make_prediction(
            model,
            scaler,
            historical_data,
            forecast_hour,
            forecast_weekday,
            forecast_month,
        )
        production = estimate_solar_production(
            num_solar_panels,
            panel_wattage,
            forecast_clouds,
            hour=forecast_hour,
            month=forecast_month,
            temp_c=forecast_temp,
        )
        forecast_price = electricity_price(
            forecast_hour,
            forecast_weekday,
            cycle,
            price_type,
        )

        balance = production - consumption
        grid_energy = max(consumption - production, 0)
        estimated_cost = grid_energy * forecast_price

        rows.append(
            {
                "Hora": f"{forecast_hour:02d}:00",
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

    return pd.DataFrame(rows)


def build_energy_frame(
    model: Any,
    scaler: Any,
    historical_data: Any,
    hour: int,
    weekday: int,
    month: int,
    cycle: str,
    price_type: str,
    num_solar_panels: int,
    panel_wattage: int,
    weather_clouds_by_hour: dict[int, float],
    weather_temp_by_hour: dict[int, float | None],
    fallback_clouds: float,
    fallback_temp: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for i in range(24):
        h = (hour + i) % 24
        d = (weekday + ((hour + i) // 24)) % 7
        consumption_h = make_prediction(model, scaler, historical_data, h, d, month)
        clouds_h = weather_clouds_by_hour.get(h, fallback_clouds)
        temp_h = weather_temp_by_hour.get(h, fallback_temp)
        production_h = estimate_solar_production(
            num_solar_panels,
            panel_wattage,
            clouds_h,
            hour=h,
            temp_c=temp_h,
        )
        rows.append(
            {
                "Hour": h,
                "Consumo_previsto": consumption_h,
                "Producao_solar": production_h,
                "Balanco": production_h - consumption_h,
                "Clouds": clouds_h,
                "Temp": temp_h,
                "Preco": electricity_price(h, weekday, cycle, price_type),
            }
        )

    df_energy = pd.DataFrame(rows)
    df_energy["Custo_liquido"] = (
        (df_energy["Consumo_previsto"] - df_energy["Producao_solar"])
        .clip(lower=0)
        * df_energy["Preco"]
    )
    return df_energy


def render_current_tab(
    t: dict[str, Any],
    weather: dict[str, Any] | None,
    price: float,
    predicted_consumption: float,
    predicted_production: float,
    energy_balance: float,
    production_24h: float,
    city: str,
    now: datetime,
    cloud_coverage: float,
) -> None:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            t["temp"],
            f"{weather['temperature']} C" if weather else "N/A",
            delta=f"Humidade: {weather['humidity']}%" if weather else None,
        )
    with col2:
        st.metric(t["price"], f"{price:.2f} €/kWh")
    with col3:
        st.metric(t["clouds_label"], f"{weather['clouds']}%" if weather else "N/A")
    with col4:
        wind_kmh = weather["wind_speed"] * 3.6 if weather else None
        st.metric(t["wind_label"], f"{wind_kmh:.1f} km/h" if wind_kmh is not None else "N/A")

    if energy_balance > 0.1:
        balance_text = "Excedente solar disponivel"
    elif energy_balance >= -0.1:
        balance_text = "Consumo quase totalmente coberto"
    else:
        balance_text = "Necessario importar energia da rede"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Consumo previsto", f"{predicted_consumption:.2f} kWh")
    with c2:
        st.metric("Producao solar agora", f"{predicted_production:.2f} kWh")
    with c3:
        st.metric("Balanco atual", f"{energy_balance:.2f} kWh")

    st.metric("Producao solar prevista (24h)", f"{production_24h:.2f} kWh")

    st.caption(balance_text)
    if weather:
        st.caption(
            f"{weather['temperature']:.1f} C | {weather.get('weather_description', 'sem descricao')} | "
            f"Vento {weather['wind_speed'] * 3.6:.1f} km/h | Nuvens {cloud_coverage}%"
        )
    else:
        st.caption(f"Meteorologia indisponivel para {city}")

    st.info(
        "Este resumo junta previsao de consumo, producao solar e balanco energetico atual."
    )


def render_forecast_tab(
    df_forecast: pd.DataFrame,
    t: dict[str, Any],
) -> None:
    st.subheader(t["forecast_section"])
    st.caption(
        "A previsao meteorologica e mostrada por hora e cruzada com o consumo previsto."
    )

    if df_forecast.empty:
        st.warning("Nao foi possivel obter previsoes para hoje.")
        return

    melhor_periodo = df_forecast.loc[df_forecast["Custo estimado (€)"].idxmin()]
    st.success(
        f"Melhor hora para consumos flexiveis: {melhor_periodo['Hora']} | "
        f"Custo estimado na rede: {melhor_periodo['Custo estimado (€)']:.3f} EUR"
    )

    st.dataframe(df_forecast, use_container_width=True, hide_index=True)

    fig_forecast = px.line(
        df_forecast,
        x="Hora",
        y=["Consumo previsto (kWh)", "Produção solar (kWh)"],
        markers=True,
        title="Consumo previsto vs. producao solar ao longo do dia",
    )
    st.plotly_chart(fig_forecast, use_container_width=True)


def render_simulator_tab(
    df_energy: pd.DataFrame,
    t: dict[str, Any],
    price: float,
    cloud_coverage: float,
    weather: dict[str, Any] | None,
    city: str,
    now: datetime,
) -> tuple[float, float, float]:
    st.subheader("Simulador e poupança")

    custo_sem_solar = 0.0
    custo_com_solar = 0.0

    for _, row in df_energy.iterrows():
        consumo_h = float(row["Consumo_previsto"])
        producao_h = float(row["Producao_solar"])
        preco_h = float(row["Preco"])
        custo_sem_solar += consumo_h * preco_h
        custo_com_solar += max(consumo_h - producao_h, 0.0) * preco_h

    poupanca = custo_sem_solar - custo_com_solar

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Custo sem solar (24h)", f"{custo_sem_solar:.2f} EUR")
    with c2:
        st.metric("Custo com solar (24h)", f"{custo_com_solar:.2f} EUR")
    with c3:
        st.metric("Poupanca estimada (24h)", f"{poupanca:.2f} EUR")

    fig_energy = px.line(
        df_energy,
        x="Hour",
        y=["Consumo_previsto", "Producao_solar"],
        markers=True,
        title="Consumo vs Producao (24h)",
    )
    fig_energy.add_vline(x=now.hour, line_dash="dash", line_color="red", annotation_text="Agora")
    st.plotly_chart(fig_energy, use_container_width=True)



    return custo_sem_solar, custo_com_solar, poupanca


def render_settings_tab(city: str, cycle: str, price_type: str, num_solar_panels: int, panel_wattage: int) -> None:
    st.subheader("Configuracoes atuais")
    st.write(f"Cidade: {city}")
    st.write(f"Ciclo: {cycle}")
    st.write(f"Tipo de preco: {price_type}")
    st.write(f"Paineis solares: {num_solar_panels}")
    st.write(f"Potencia por painel: {panel_wattage} W")

    st.caption(
        "Se quiseres, podes usar esta tab para confirmar o contexto antes de alterar o ficheiro original."
    )


init_session_state()
model, scaler, historical_data = load_all()
config = get_config()
t = PT_TEXTS

if st.session_state.abrir_config and not dialog_opened:
    if st.session_state.get("user") is None:
        popup_login()
        dialog_opened = True
    else:
        st.session_state.abrir_config = False
        popup_configuracao()
        dialog_opened = True

config = get_config()
city = config["city"]
cycle = config["cycle"]
price_type = config["price_type"]
num_solar_panels = int(config["num_solar_panels"])
panel_wattage = int(config["panel_wattage"])

now = datetime.now()
hour = now.hour
weekday = now.weekday()
month = now.month

weather = weather_for_city(city)
cloud_coverage = weather["clouds"] if weather else 0
current_temp = weather.get("temperature") if weather else None

price = electricity_price(hour, weekday, cycle, price_type)
predicted_consumption = make_prediction(model, scaler, historical_data, hour, weekday, month)
predicted_production = estimate_solar_production(
    num_solar_panels,
    panel_wattage,
    cloud_coverage,
    hour=hour,
    temp_c=current_temp,
)
energy_balance = predicted_production - predicted_consumption

weather_forecast = future_weather(city)
forecast_clouds_by_hour: dict[int, float] = {}
forecast_temp_by_hour: dict[int, float | None] = {}

weather_hourly_lookup = build_hourly_weather_lookup(
    weather_forecast,
    now,
    cloud_coverage,
    current_temp,
)
for forecast_hour, data in weather_hourly_lookup.items():
    forecast_clouds_by_hour[forecast_hour] = float(data["clouds"])
    forecast_temp_by_hour[forecast_hour] = data["temp"]

forecast_today = build_current_forecast_frame(
    weather_hourly_lookup,
    model,
    scaler,
    historical_data,
    cycle,
    price_type,
    num_solar_panels,
    panel_wattage,
)

energy_24h = build_energy_frame(
    model,
    scaler,
    historical_data,
    hour,
    weekday,
    month,
    cycle,
    price_type,
    num_solar_panels,
    panel_wattage,
    forecast_clouds_by_hour,
    forecast_temp_by_hour,
    cloud_coverage,
    current_temp,
)

render_header(t, city, now)

if predicted_production > predicted_consumption:
    status_line = "A producao solar esta a cobrir parte do consumo."
elif price <= 0.15:
    status_line = "Preco em vazio neste momento."
elif price > 0.22 and predicted_production < predicted_consumption:
    status_line = "Estes num periodo tarifario caro."
elif cloud_coverage > 70:
    status_line = "Producao solar limitada pelas nuvens."
else:
    status_line = "Este e um bom momento para utilizar eletrodomesticos."

st.info(status_line)


current_tab, forecast_tab, simulator_tab, settings_tab = st.tabs(
    ["Resumo", "Previsao do dia", "Simulador", "Configuracoes"]
)

with current_tab:
    producao_24h_total = float(energy_24h["Producao_solar"].sum())

    render_current_tab(
        t=t,
        weather=weather,
        price=price,
        predicted_consumption=predicted_consumption,
        predicted_production=predicted_production,
        energy_balance=energy_balance,
        production_24h=producao_24h_total,
        city=city,
        now=now,
        cloud_coverage=cloud_coverage,
    )

    advices = generate_recommendation(
        predicted_consumption,
        predicted_production,
        price,
        cloud_coverage,
        t,
    )
    st.subheader(t["rec_title"])
    for advice in advices:
        if advice in {t["rec_low_price"], t["rec_high_solar"]}:
            st.success(advice)
        elif advice in {t["rec_high_price"], t["rec_partial_solar"], t["rec_clouds"]}:
            st.warning(advice)
        else:
            st.info(advice)

    st.subheader(t["hourly_chart"])
    prices = [electricity_price(h, weekday, cycle, price_type) for h in range(24)]
    df_tariff = pd.DataFrame({"Hour": list(range(24)), "Price (€/kWh)": prices})
    fig_price = px.line(
        df_tariff,
        x="Hour",
        y="Price (€/kWh)",
        markers=True,
        title=t["chart_title"],
        color_discrete_sequence=["#f7a600"],
    )
    fig_price.add_vline(x=hour, line_dash="dash", line_color="red", annotation_text=t["now_label"])
    st.plotly_chart(fig_price, use_container_width=True)





with forecast_tab:
    render_forecast_tab(forecast_today, t)


with simulator_tab:
    custo_sem_solar, custo_com_solar, poupanca = render_simulator_tab(
        energy_24h,
        t,
        price,
        cloud_coverage,
        weather,
        city,
        now,
    )

    with st.expander("Comparar ciclos tarifarios"):
        rows_compare = []
        for cycle_name in ["Simples", "Bi-horária", "Tri-horária"]:
            for h in range(24):
                rows_compare.append(
                    {
                        "Hour": h,
                        "Price": electricity_price(h, weekday, cycle_name, price_type),
                        "Cycle": cycle_name,
                    }
                )
        df_compare = pd.DataFrame(rows_compare)
        fig_compare = px.line(
            df_compare,
            x="Hour",
            y="Price",
            color="Cycle",
            markers=True,
            title="Comparacao de ciclos tarifarios",
        )
        fig_compare.add_vline(x=hour, line_dash="dash", line_color="red", annotation_text="Agora")
        st.plotly_chart(fig_compare, use_container_width=True)

    consumo_24h = float(energy_24h["Consumo_previsto"].sum())
    producao_24h = float(energy_24h["Producao_solar"].sum())
    balanco_24h = producao_24h - consumo_24h

    simulation_payload = {
        "city": city,
        "cycle": cycle,
        "price_type": price_type,
        "num_solar_panels": num_solar_panels,
        "panel_wattage": panel_wattage,
        "predicted_consumption": consumo_24h,
        "predicted_production": producao_24h,
        "energy_balance": balanco_24h,
        "price_now": float(price),
        "estimated_cost_without_solar": float(custo_sem_solar),
        "estimated_cost_with_solar": float(custo_com_solar),
        "estimated_savings": float(poupanca),
        "model_version": "xgboost_v1",
    }

    scenario_name = st.text_input(
        "Nome do cenário",
        value="A minha simulação",
        help="Exemplo: Apartamento em Lisboa com 6 painéis.",
    )

    if st.button("Guardar simulação no histórico", width="stretch"):
        if st.session_state.get("user") is None:
            st.warning("Tens de iniciar sessão para guardar cenários.")
        else:
            payload = {
                **simulation_payload,
                "client_id": st.session_state.user.id,
                "scenario_name": scenario_name,
            }
            
            # Mostra um spinner enquanto grava
            with st.spinner("A guardar..."):
                ok, msg = save_simulation(payload)
            
            if ok:
                st.toast("Cenário guardado com sucesso!")
                st.success("Simulação registada no teu histórico.")
            else:
                st.error(msg)

with settings_tab:
    render_settings_tab(city, cycle, price_type, num_solar_panels, panel_wattage)

with st.bottom:
    st.caption("Smart Energy Advisor © 2026 | Powered by XGBoost + OpenWeatherMap")
