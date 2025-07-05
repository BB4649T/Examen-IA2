import requests
from datetime import datetime
import logging
import time  # Ajoute cette ligne avec les autres imports
from typing import Dict, List
from utils import (
    OPENWEATHER_API_KEY, BASE_URL, CITIES, 
    save_data, validate_api_response, retry_request, 
    kelvin_to_celsius, setup_logging
)

logger = setup_logging()

def extract_current_weather(city: Dict) -> Dict:
    """Extrait les données météo actuelles pour une ville"""
    
    def make_request():
        url = f"{BASE_URL}/weather"
        params = {
            'lat': city['lat'],
            'lon': city['lon'],
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params)
        if not validate_api_response(response):
            raise Exception(f"Erreur API pour {city['name']}")
        
        return response.json()
    
    try:
        data = retry_request(make_request)
        
        # Transformation des données
        weather_data = {
            'city': city['name'],
            'country': city['country'],
            'lat': city['lat'],
            'lon': city['lon'],
            'timestamp': datetime.now().isoformat(),
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data.get('wind', {}).get('speed', 0),
            'wind_direction': data.get('wind', {}).get('deg', 0),
            'cloudiness': data['clouds']['all'],
            'visibility': data.get('visibility', 0) / 1000,  # En km
            'weather_main': data['weather'][0]['main'],
            'weather_description': data['weather'][0]['description'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).isoformat(),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).isoformat(),
            'data_source': 'current_weather'
        }
        
        logger.info(f"Données extraites pour {city['name']}: {weather_data['temperature']}°C")
        return weather_data
        
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction pour {city['name']}: {e}")
        return None

def extract_forecast(city: Dict, days: int = 5) -> List[Dict]:
    """Extrait les prévisions météo pour une ville"""
    
    def make_request():
        url = f"{BASE_URL}/forecast"
        params = {
            'lat': city['lat'],
            'lon': city['lon'],
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric',
            'cnt': days * 8  # 8 prévisions par jour (toutes les 3h)
        }
        
        response = requests.get(url, params=params)
        if not validate_api_response(response):
            raise Exception(f"Erreur API prévisions pour {city['name']}")
        
        return response.json()
    
    try:
        data = retry_request(make_request)
        forecasts = []
        
        for forecast in data['list']:
            forecast_data = {
                'city': city['name'],
                'country': city['country'],
                'lat': city['lat'],
                'lon': city['lon'],
                'forecast_time': datetime.fromtimestamp(forecast['dt']).isoformat(),
                'temperature': forecast['main']['temp'],
                'feels_like': forecast['main']['feels_like'],
                'humidity': forecast['main']['humidity'],
                'pressure': forecast['main']['pressure'],
                'wind_speed': forecast.get('wind', {}).get('speed', 0),
                'wind_direction': forecast.get('wind', {}).get('deg', 0),
                'cloudiness': forecast['clouds']['all'],
                'weather_main': forecast['weather'][0]['main'],
                'weather_description': forecast['weather'][0]['description'],
                'rain_3h': forecast.get('rain', {}).get('3h', 0),
                'snow_3h': forecast.get('snow', {}).get('3h', 0),
                'data_source': 'forecast'
            }
            forecasts.append(forecast_data)
        
        logger.info(f"Prévisions extraites pour {city['name']}: {len(forecasts)} points")
        return forecasts
        
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des prévisions pour {city['name']}: {e}")
        return []

def extract_air_quality(city: Dict) -> Dict:
    """Extrait les données de qualité de l'air"""
    
    def make_request():
        url = f"{BASE_URL}/air_pollution"
        params = {
            'lat': city['lat'],
            'lon': city['lon'],
            'appid': OPENWEATHER_API_KEY
        }
        
        response = requests.get(url, params=params)
        if not validate_api_response(response):
            raise Exception(f"Erreur API qualité air pour {city['name']}")
        
        return response.json()
    
    try:
        data = retry_request(make_request)
        
        air_quality = {
            'city': city['name'],
            'country': city['country'],
            'timestamp': datetime.now().isoformat(),
            'aqi': data['list'][0]['main']['aqi'],  # Air Quality Index
            'co': data['list'][0]['components']['co'],
            'no2': data['list'][0]['components']['no2'],
            'o3': data['list'][0]['components']['o3'],
            'pm2_5': data['list'][0]['components']['pm2_5'],
            'pm10': data['list'][0]['components']['pm10'],
            'data_source': 'air_quality'
        }
        
        logger.info(f"Qualité de l'air extraite pour {city['name']}: AQI {air_quality['aqi']}")
        return air_quality
        
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction qualité air pour {city['name']}: {e}")
        return None

def run_current_extraction():
    """Exécute l'extraction des données actuelles pour toutes les villes"""
    logger.info("Début de l'extraction des données actuelles")
    
    current_data = []
    forecast_data = []
    air_quality_data = []
    
    for city in CITIES:
        # Données actuelles
        current = extract_current_weather(city)
        if current:
            current_data.append(current)
        
        # Prévisions
        forecasts = extract_forecast(city)
        forecast_data.extend(forecasts)
        
        # Qualité de l'air
        air_quality = extract_air_quality(city)
        if air_quality:
            air_quality_data.append(air_quality)
    
    # Sauvegarde
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_data(current_data, f"current_weather_{timestamp}.json")
    save_data(forecast_data, f"forecast_weather_{timestamp}.json")
    save_data(air_quality_data, f"air_quality_{timestamp}.json")
    
    logger.info(f"Extraction terminée: {len(current_data)} villes, {len(forecast_data)} prévisions")
    
    return {
        'current_file': f"current_weather_{timestamp}.json",
        'forecast_file': f"forecast_weather_{timestamp}.json",
        'air_quality_file': f"air_quality_{timestamp}.json"
    }

if __name__ == "__main__":
    run_current_extraction()