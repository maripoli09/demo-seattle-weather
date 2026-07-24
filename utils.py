import pandas as pd
import pickle
import joblib
import numpy as np
import math
from datetime import datetime, timedelta


FEATURE_COLS = ["hour", "day_of_week", "month", "is_weekend", "lag_1", "lag_48"]
_DEFAULT_INPUT_MODE = "scaled"


def _safe_load_pkl(path):
    """Load .pkl files serialized either with pickle or joblib."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(path)

def load_smart_models():
    """
    Load the models XGBoost and Scaler.

    """
    try:
        model = _safe_load_pkl('smart_energy_model.pkl')
        scaler = _safe_load_pkl('scaler.pkl')
        historical_data = _safe_load_pkl('df_gc_clean.pkl')
        return model, scaler, historical_data

    except Exception as e:
        print(f"Error loading .pkl files: {e}")
        return None, None, None


def _mae(y_true, y_pred):
    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true_np - y_pred_np)))


def _predict_with_mode(model, scaler, X: pd.DataFrame, mode: str):
    if mode == "scaled" and scaler is not None:
        return model.predict(scaler.transform(X))
    return model.predict(X)


def resolve_model_input_mode(model, scaler, historical_data) -> str:
    """
    Detect whether the serialized model expects scaled or raw features.

    We score both paths on a recent holdout slice from historical_data and keep
    the one with lower MAE. Result is cached in DataFrame attrs.
    """
    if historical_data is None or not isinstance(historical_data, pd.DataFrame):
        return _DEFAULT_INPUT_MODE

    cached_mode = historical_data.attrs.get("_model_input_mode")
    if cached_mode in {"scaled", "raw"}:
        return cached_mode

    required_cols = FEATURE_COLS + ["energy_kwh"]
    if any(col not in historical_data.columns for col in required_cols):
        historical_data.attrs["_model_input_mode"] = _DEFAULT_INPUT_MODE
        return _DEFAULT_INPUT_MODE

    df_eval = historical_data.dropna(subset=required_cols).copy()
    if len(df_eval) < 200:
        historical_data.attrs["_model_input_mode"] = _DEFAULT_INPUT_MODE
        return _DEFAULT_INPUT_MODE

    split_point = int(len(df_eval) * 0.8)
    test = df_eval.iloc[split_point:].tail(min(1000, max(100, len(df_eval) - split_point)))
    if test.empty:
        historical_data.attrs["_model_input_mode"] = _DEFAULT_INPUT_MODE
        return _DEFAULT_INPUT_MODE

    X_eval = test[FEATURE_COLS]
    y_eval = test["energy_kwh"]

    mae_scaled = float("inf")
    mae_raw = float("inf")

    try:
        pred_scaled = np.clip(_predict_with_mode(model, scaler, X_eval, "scaled"), a_min=0, a_max=None)
        mae_scaled = _mae(y_eval, pred_scaled)
    except Exception:
        pass

    try:
        pred_raw = np.clip(_predict_with_mode(model, scaler, X_eval, "raw"), a_min=0, a_max=None)
        mae_raw = _mae(y_eval, pred_raw)
    except Exception:
        pass

    mode = "scaled" if mae_scaled <= mae_raw else "raw"
    historical_data.attrs["_model_input_mode"] = mode
    return mode


def predict_consumption_batch(model, scaler, X: pd.DataFrame, historical_data) -> np.ndarray:
    """Predict using the automatically resolved input mode for the current model."""
    mode = resolve_model_input_mode(model, scaler, historical_data)
    preds = _predict_with_mode(model, scaler, X, mode)
    return np.clip(np.asarray(preds, dtype=float), a_min=0, a_max=None)
    
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


    features = features[FEATURE_COLS]
    prediction = predict_consumption_batch(model, scaler, features, historical_data)

    return float(max(0, prediction[0]))


def make_recursive_predictions(model, scaler, historical_data, timeline):
    """
    Recursive multi-step consumption forecast.

    Each new prediction is appended to the working history and reused as lag_1
    for the next step, producing a more realistic hour-to-hour curve.
    """
    if model is None or scaler is None or historical_data is None or len(timeline) == 0:
        return [0.0] * len(timeline)

    if "energy_kwh" not in historical_data.columns:
        return [0.0] * len(timeline)

    history_values = historical_data["energy_kwh"].dropna().tolist()
    if not history_values:
        return [0.0] * len(timeline)

    predictions = []
    mode = resolve_model_input_mode(model, scaler, historical_data)

    for hour, day_of_week, month in timeline:
        lag_1 = history_values[-1]
        lag_48 = history_values[-48] if len(history_values) >= 48 else lag_1

        features = pd.DataFrame([
            {
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": 1 if day_of_week >= 5 else 0,
                "lag_1": lag_1,
                "lag_48": lag_48,
            }
        ])[FEATURE_COLS]

        prediction = float(max(0, _predict_with_mode(model, scaler, features, mode)[0]))

        predictions.append(prediction)
        history_values.append(prediction)

    return predictions


def estimate_solar_production(
    num_panels,
    panel_wattage,
    clouds,
    hour=None,
    month=None,
    temp_c=None,
):
    """
    Estima produção solar por hora (kWh) para um sistema residencial.
    """
    if num_panels <= 0 or panel_wattage <= 0:
        return 0.0

    now = datetime.now()
    if hour is None:
        hour = now.hour
    if month is None:
        month = now.month

    # Potência instalada (kW)
    capacity_kw = (num_panels * panel_wattage) / 1000.0

    # Janela solar aproximada para PT
    if hour < 6 or hour > 20:
        return 0.0

    # Curva diária (pico perto do meio-dia)
    daylight_curve = max(0.0, math.sin(math.pi * (hour - 6) / 14)) ** 1.35

    # Fator de nuvens (não linear, mais realista)
    cloud_pct = max(0.0, min(100.0, float(clouds)))
    cloud_factor = max(0.05, 1.0 - (cloud_pct / 100.0) ** 1.25)

    # Fator sazonal simples (Portugal)
    monthly_factor = {
        1: 0.55, 2: 0.65, 3: 0.80, 4: 0.95,
        5: 1.05, 6: 1.10, 7: 1.12, 8: 1.05,
        9: 0.90, 10: 0.75, 11: 0.60, 12: 0.50,
    }.get(month, 0.85)

    # Perdas típicas de sistema (inversor, cabos, sujidade, etc.)
    performance_ratio = 0.80

    # Temperatura: acima de 25C reduz ligeiramente a eficiência
    if temp_c is None:
        temp_factor = 1.0
    else:
        temp_factor = 1.0 - max(0.0, float(temp_c) - 25.0) * 0.004
        temp_factor = max(0.85, min(1.05, temp_factor))

    production_kwh = (
        capacity_kw
        * daylight_curve
        * cloud_factor
        * monthly_factor
        * performance_ratio
        * temp_factor
    )
    return float(max(0.0, production_kwh))

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