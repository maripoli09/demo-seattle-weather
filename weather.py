import requests

def obtain_local_weather(city="Lisboa", api_key="da106eb8a1c706a86299c4ffc3aebba3"):
    """
    Obtain weather information for a specific city using the OpenWeatherMap API. Obtém informações meteorológicas para uma cidade específica usando a API OpenWeatherMap.

    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=pt"
    
    try:
        response = requests.get(url)
        # Verefy if the request was successful (status code 200)
        response.raise_for_status()  

        data = response.json()
        
        climate = {
            "city": data.get("name"),
            "temperature": data.get("main", {}).get("temp"),
            "humidity": data.get("main", {}).get("humidity"),
            "clouds": data.get("clouds", {}).get("all"),
            "weather_description": data.get("weather", [{}])[0].get("description"),
            "icon": data.get("weather", [{}])[0].get("icon"),
            "wind_speed": data.get("wind", {}).get("speed"),
        }
        return climate
    
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to OpenWeatherMap API: {e}")
        return None
    
def future_weather(city="Lisboa", api_key="da106eb8a1c706a86299c4ffc3aebba3"):
    """
    Obtain weather forecasts for a specific city using the OpenWeatherMap API. Obtém previsões meteorológicas para uma cidade específica usando a API OpenWeatherMap.
    """
    # Note:Some account plans use the /forecast endpoint. 
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric&lang=pt"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  
        data = response.json()
        
        # Returns the list of forecasts (usually every 3 hours)
        return data["list"]
    
    except Exception as e:
        print(f"Error obtaining weather forecast: {e}")
        return None
    
if __name__ == "__main__":

    my_key = "da106eb8a1c706a86299c4ffc3aebba3"

    city = input("Indicate which city you are in: ")

    results = obtain_local_weather(city, my_key)
    
    if results:
        print(f"Climate in {results['city']}: {results['temperature']}ºC, {results['weather_description']}")
        print(f"Humidity: {results['humidity']}%")