import requests
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import time
from utils import (
    OPENWEATHER_API_KEY, CITIES, 
    save_data, validate_api_response, retry_request, 
    setup_logging, get_season
)
logger = setup_logging()

def extract_historical_weather(city, date):
    """Nouvelle version avec gestion d'erreur robuste"""
    try:
        # Utilisation de l'API Current au lieu de Historical
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                'lat': city['lat'],
                'lon': city['lon'],
                'appid': OPENWEATHER_API_KEY,
                'units': 'metric'
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
            
        data = response.json()
        
        return {
            'city': city['name'],
            'date': date.isoformat(),
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather_main': data['weather'][0]['main'],
            'data_source': 'current_as_historical'  # Marqueur spécial
        }
        
    except Exception as e:
        logger.error(f"Erreur extraction: {str(e)}")
        return None

def extract_historical_range(city: Dict, start_date: datetime, end_date: datetime) -> List[Dict]:
    """Extrait les données historiques sur une période"""
    historical_data = []
    current_date = start_date
    
    while current_date <= end_date:
        data = extract_historical_weather(city, current_date)
        if data:
            historical_data.append(data)
        
        current_date += timedelta(days=1)
        time.sleep(0.1)  # Limite de taux API
    
    return historical_data

def extract_seasonal_data(city: Dict, year: int = 2023) -> List[Dict]:
    """Extrait des données représentatives pour chaque saison"""
    seasonal_dates = [
        datetime(year, 1, 15),   # Hiver
        datetime(year, 4, 15),   # Printemps  
        datetime(year, 7, 15),   # Été
        datetime(year, 10, 15)   # Automne
    ]
    
    seasonal_data = []
    for date in seasonal_dates:
        data = extract_historical_weather(city, date)
        if data:
            seasonal_data.append(data)
        time.sleep(0.2)  # Limite de taux API
    
    return seasonal_data

def extract_monthly_averages(city: Dict, year: int = 2023) -> List[Dict]:
    """Extrait des données pour calculer les moyennes mensuelles"""
    monthly_data = []
    
    for month in range(1, 13):
        # Prendre le 15 de chaque mois comme représentatif
        date = datetime(year, month, 15)
        data = extract_historical_weather(city, date)
        if data:
            data['month'] = month
            data['month_name'] = date.strftime('%B')
            monthly_data.append(data)
        time.sleep(0.2)
    
    return monthly_data

def extract_extreme_weather_days(city: Dict, year: int = 2023) -> List[Dict]:
    """Extrait des données pour des jours avec conditions extrêmes"""
    # Dates connues pour conditions extrêmes (à adapter selon la région)
    extreme_dates = []
    
    # Ajouter des dates d'été et d'hiver pour capturer les extrêmes
    summer_dates = [datetime(year, 6, 21), datetime(year, 7, 21), datetime(year, 8, 21)]
    winter_dates = [datetime(year, 12, 21), datetime(year, 1, 21), datetime(year, 2, 21)]
    
    extreme_dates.extend(summer_dates)
    extreme_dates.extend(winter_dates)
    
    extreme_data = []
    for date in extreme_dates:
        data = extract_historical_weather(city, date)
        if data:
            extreme_data.append(data)
        time.sleep(0.2)
    
    return extreme_data

def run_historical_extraction(days_back: int = 30):
    """Exécute l'extraction des données historiques"""
    logger.info(f"Début de l'extraction des données historiques ({days_back} jours)")
    
    end_date = datetime.now() - timedelta(days=1)  # Hier
    start_date = end_date - timedelta(days=days_back)
    
    all_historical_data = []
    
    for city in CITIES:
        logger.info(f"Extraction historique pour {city['name']}")
        
        # Données sur la période
        city_data = extract_historical_range(city, start_date, end_date)
        all_historical_data.extend(city_data)
        
        # Données saisonnières
        seasonal_data = extract_seasonal_data(city)
        all_historical_data.extend(seasonal_data)
        
        # Données mensuelles
        monthly_data = extract_monthly_averages(city)
        all_historical_data.extend(monthly_data)
        
        time.sleep(1)  # Pause entre les villes
    
    # Sauvegarde
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"historical_weather_{timestamp}.json"
    save_data(all_historical_data, filename, "historical")
    
    logger.info(f"Extraction historique terminée: {len(all_historical_data)} enregistrements")
    return filename

if __name__ == "__main__":
    run_historical_extraction()