import pandas as pd
import pickle
import numpy as np
import math
from datetime import datetime, timedelta

def load_smart_models():
    """
    Load the models XGBoost and Scaler.

    """
    try:
        with open('smart_energy_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as s:
            scaler = pickle.load(s)
        with open('df_gc_clean.pkl', 'rb') as d:
            historical_data = pickle.load(d)
        return model, scaler, historical_data

    except Exception as e:
        print(f"Error loading .pkl files: {e}")
        return None, None, None
    
def make_prediction(model, scaler, historical_data, hour, day_of_week, month):
    """
    Does the prediction using the columns of the notebook
    """
    if model is None or scaler is None or historical_data is None:
        return 0.0

    lag_1 = historical_data['energy_kwh'].iloc[-1]
    lag_48 = historical_data['energy_kwh'].iloc[-48] if len(historical_data) >= 48 else lag_1

    features = pd.DataFrame([{
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': 1 if day_of_week >= 5 else 0,
        'lag_1': lag_1,
        'lag_48': lag_48
    }])


    cols_order =['hour', 'day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_48']
    features = features[cols_order]

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    return float(max(0, prediction[0]))


def estimate_solar_production(num_panels, panel_wattage, clouds, hour=None):
    """
    Estimate of solar production based in the infrastructure of the user and clouds.
    """
    if num_panels <= 0:
        return 0.0
    
    if hour is None:
        hour = datetime.now().hour
    
    # Maximum installed power in kW
    max_power_kw = (num_panels * panel_wattage) / 1000.0

    # Cloud loss factor(simplified)
    # If clouds=0%, fator=1.0. If clouds=100%, fator=0.2
    cloud_factor = 1.0 - (0.8 * (clouds / 100.0))
    cloud_factor = max(0.2, min(1.0, cloud_factor))


    # Ajusted by hour of the day (production just between 7h and 20h)
    if 7 <= hour <= 20:
        time_factor = math.sin(math.pi * (hour - 7)/13)

        return float(max_power_kw * cloud_factor * max(0.0, time_factor))
    return 0.0


def generate_recommendation(predicted_consumption, predicted_production, current_price, cloud_coverage, t):
    """
    Generate recommendation based on predicted consumption, production, current price, and cloud coverage.

    """
    advices = []

    if current_price <= 0.15:
        advices.append(t["rec_low_price"])
    elif current_price > 0.22:
        advices.append(t["rec_high_price"])
    
    if predicted_production > predicted_consumption:
        advices.append(t["rec_high_solar"])
    elif predicted_production > 0 and predicted_consumption > 0:
        advices.append(t["rec_partial_solar"])

    if cloud_coverage > 70:
        advices.append(t["rec_clouds"])

    return advices if advices else [t["rec_none"]]


# translations dictionary
LANGUAGES = {
    "PT": {
        "temp": "Temperatura",
        "price": "Preço Atual",
        "hourly_chart": "Horário Tarifário",
        "forecast_section": "Previsão de Consumo (IA)",
        "prediction_text": "Consumo previsto para agora",
        "solar_label": "Produção Solar Prevista",
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
        "definitions": "Definições",
        "language_label": "Lingua",
        "location_label": "A minha Localização",
        "city_label": "Cidade",
        "tariff_label": "Tarifa",
        "cycle_label": "Ciclo",
        "cycle_options": ["Simples", "Bi-horária", "Tri-horária"],
        "price_label": "Tipo de preço",
        "price_options": ["Preço fixo", "Preço variável"],
        "panel_number_label": "Número de Painéis Solares",
        "panel_power_label": "Potência de cada Painel (W)",
        "installed_power_label": "Potência Instalada",
    },
    "EN": {
        "temp": "Temperature",
        "price": "Current Price",
        "hourly_chart": "Hourly Tariff Schedule",
        "forecast_section": "Consumption Forecast (AI)",
        "prediction_text": "Predicted consumption for now",
        "solar_label": "Predicted Solar Production",
        "clouds_label": "Cloud Coverage",
        "wind_label": "Wind",
        "rec_title": "Recommendations",
        "footer": "Updated at",
        "now_label": "Now",
        "chart_title": "Electricity Price Throughout the Day",
        "rec_low_price": "Low price — Good moment to use electricity.",
        "rec_high_price": "High price — Consider reducing electricity usage.",
        "rec_high_solar": "High solar production — Recommend using solar energy.",
        "rec_partial_solar": "Solar production is active but insufficient — Try reducing usage to maximize self-consumption.",
        "rec_clouds": "High cloud coverage — Solar production may be low.",
        "rec_none": "No specific advice at this moment.",
        "definitions": "Definitions",
        "language_label": "Language",
        "location_label": "My Location",
        "city_label": "City",
        "tariff_label": "Tariff",
        "cycle_label": "Cycle",
        "cycle_options": ["Simple", "Two-cycle", "Three-cycle"],
        "price_label": "Price Type",
        "price_options": ["Fixed price", "Variable price"],
        "panel_number_label": "Number of Solar Panels",
        "panel_power_label": "Power of each Panel (W)",
        "installed_power_label": "Installed Power",
    }
}