import requests
import streamlit as st

def obtain_local_weather(city="Lisboa"):
    """
    Obtain weather information for a specific city using the OpenWeatherMap API. Obtém informações meteorológicas para uma cidade específica usando a API OpenWeatherMap.

    """
    api_key = st.secrets.get("OPENWEATHER_API_KEY")
    if not api_key:
        return None

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=pt"
    
    try:
        response = requests.get(url, timeout=10)
        # Verefy if the request was successful (status code 200)
        response.raise_for_status()  

        data = response.json()
        
        return {
            "city": data.get("name"),
            "temperature": data.get("main", {}).get("temp"),
            "humidity": data.get("main", {}).get("humidity"),
            "clouds": data.get("clouds", {}).get("all"),
            "weather_description": data.get("weather", [{}])[0].get("description"),
            "icon": data.get("weather", [{}])[0].get("icon"),
            "wind_speed": data.get("wind", {}).get("speed"),
        }
    
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to OpenWeatherMap API: {e}")
        return None
    
def future_weather(city="Lisboa"):
    """
    Obtain weather forecasts for a specific city using the OpenWeatherMap API. Obtém previsões meteorológicas para uma cidade específica usando a API OpenWeatherMap.
    """
    api_key = st.secrets.get("OPENWEATHER_API_KEY")
    if not api_key:
        return None

    # Note:Some account plans use the /forecast endpoint. 
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric&lang=pt"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  
        
        # Returns the list of forecasts (usually every 3 hours)
        return response.json()["list"]
    
    except Exception as e:
        st.warning(f"Erro ao obter previsão: {e}")
        return None
    
