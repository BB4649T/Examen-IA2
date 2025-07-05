import os
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import numpy as np

# Configuration
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', 'c26b1618fd191d81352684485abb8720')
BASE_URL = "https://api.openweathermap.org/data/2.5"
HISTORICAL_URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
DATA_DIR = "/tmp/weather_data"

# Liste des villes à analyser
CITIES = [
    {"name": "Paris", "lat": 48.8566, "lon": 2.3522, "country": "FR"},
    {"name": "London", "lat": 51.5074, "lon": -0.1278, "country": "GB"},
    {"name": "New York", "lat": 40.7128, "lon": -74.0060, "country": "US"},
    {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503, "country": "JP"},
    {"name": "Sydney", "lat": -33.8688, "lon": 151.2093, "country": "AU"},
    {"name": "Cairo", "lat": 30.0444, "lon": 31.2357, "country": "EG"},
    {"name": "Moscow", "lat": 55.7558, "lon": 37.6176, "country": "RU"},
    {"name": "Rio de Janeiro", "lat": -22.9068, "lon": -43.1729, "country": "BR"},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "country": "IN"},
    {"name": "Cape Town", "lat": -33.9249, "lon": 18.4241, "country": "ZA"}
]

def setup_logging():
    """Configuration des logs"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_data_directory():
    """Crée le répertoire de données si nécessaire"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(f"{DATA_DIR}/raw", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/processed", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/historical", exist_ok=True)

def save_data(data: Dict, filename: str, subfolder: str = "raw"):
    """Sauvegarde des données en JSON"""
    filepath = f"{DATA_DIR}/{subfolder}/{filename}"
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    return filepath

def load_data(filename: str, subfolder: str = "raw") -> Dict:
    """Chargement des données depuis JSON"""
    filepath = f"{DATA_DIR}/{subfolder}/{filename}"
    with open(filepath, 'r') as f:
        return json.load(f)

def kelvin_to_celsius(kelvin: float) -> float:
    """Conversion Kelvin vers Celsius"""
    return kelvin - 273.15

def calculate_comfort_index(temp: float, humidity: float, wind_speed: float) -> float:
    """Calcul d'un indice de confort basé sur température, humidité et vent"""
    # Formule simplifiée d'indice de confort
    heat_index = temp + 0.5 * (humidity - 60) / 100 * (temp - 20)
    wind_chill = temp - 0.5 * wind_speed
    return (heat_index + wind_chill) / 2

def get_season(date: datetime, lat: float) -> str:
    """Détermine la saison basée sur la date et la latitude"""
    month = date.month
    
    # Hémisphère nord
    if lat >= 0:
        if month in [12, 1, 2]:
            return "Hiver"
        elif month in [3, 4, 5]:
            return "Printemps"
        elif month in [6, 7, 8]:
            return "Été"
        else:
            return "Automne"
    # Hémisphère sud
    else:
        if month in [12, 1, 2]:
            return "Été"
        elif month in [3, 4, 5]:
            return "Automne"
        elif month in [6, 7, 8]:
            return "Hiver"
        else:
            return "Printemps"

def validate_api_response(response: requests.Response) -> bool:
    """Valide la réponse de l'API"""
    if response.status_code == 200:
        return True
    else:
        logging.error(f"Erreur API: {response.status_code} - {response.text}")
        return False

def retry_request(func, max_retries: int = 3, delay: int = 1):
    """Fonction de retry pour les requêtes API"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            logging.warning(f"Tentative {attempt + 1} échouée: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(delay * (2 ** attempt))  # Backoff exponentiel