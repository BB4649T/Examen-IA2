import json
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Optional
import glob
import os
from utils import (
    DATA_DIR, save_data, load_data, setup_logging
)

logger = setup_logging()

def load_latest_files() -> Dict[str, str]:
    """Trouve les fichiers les plus récents de chaque type"""
    files = {
        'current': None,
        'forecast': None,
        'air_quality': None,
        'historical': None
    }
    
    # Fichiers actuels
    current_files = glob.glob(f"{DATA_DIR}/raw/current_weather_*.json")
    if current_files:
        files['current'] = max(current_files, key=os.path.getctime)
    
    # Fichiers prévisions
    forecast_files = glob.glob(f"{DATA_DIR}/raw/forecast_weather_*.json")
    if forecast_files:
        files['forecast'] = max(forecast_files, key=os.path.getctime)
    
    # Fichiers qualité air
    air_files = glob.glob(f"{DATA_DIR}/raw/air_quality_*.json")
    if air_files:
        files['air_quality'] = max(air_files, key=os.path.getctime)
    
    # Fichiers historiques
    hist_files = glob.glob(f"{DATA_DIR}/historical/historical_weather_*.json")
    if hist_files:
        files['historical'] = max(hist_files, key=os.path.getctime)
    
    return files

def merge_current_and_air_quality(current_data: List[Dict], air_quality_data: List[Dict]) -> List[Dict]:
    """Fusionne les données actuelles avec la qualité de l'air"""
    merged_data = []
    
    # Créer un dictionnaire pour un accès rapide aux données de qualité de l'air
    air_quality_dict = {item['city']: item for item in air_quality_data}
    
    for current in current_data:
        merged_item = current.copy()
        
        # Ajouter les données de qualité de l'air si disponibles
        if current['city'] in air_quality_dict:
            air_data = air_quality_dict[current['city']]
            merged_item.update({
                'aqi': air_data['aqi'],
                'co': air_data['co'],
                'no2': air_data['no2'],
                'o3': air_data['o3'],
                'pm2_5': air_data['pm2_5'],
                'pm10': air_data['pm10']
            })
        
        merged_data.append(merged_item)
    
    return merged_data

def aggregate_forecast_data(forecast_data: List[Dict]) -> List[Dict]:
    """Agrège les données de prévision par ville et jour"""
    df = pd.DataFrame(forecast_data)
    
    # Convertir les timestamps
    df['forecast_time'] = pd.to_datetime(df['forecast_time'])
    df['date'] = df['forecast_time'].dt.date
    
    # Grouper par ville et date
    daily_forecasts = []
    
    for (city, date), group in df.groupby(['city', 'date']):
        daily_forecast = {
            'city': city,
            'country': group['country'].iloc[0],
            'lat': group['lat'].iloc[0],
            'lon': group['lon'].iloc[0],
            'date': date.isoformat(),
            'temp_min': group['temperature'].min(),
            'temp_max': group['temperature'].max(),
            'temp_avg': group['temperature'].mean(),
            'humidity_avg': group['humidity'].mean(),
            'pressure_avg': group['pressure'].mean(),
            'wind_speed_avg': group['wind_speed'].mean(),
            'cloudiness_avg': group['cloudiness'].mean(),
            'rain_total': group['rain_3h'].sum(),
            'snow_total': group['snow_3h'].sum(),
            'dominant_weather': group['weather_main'].mode().iloc[0] if not group['weather_main'].mode().empty else 'Unknown',
            'data_source': 'forecast_aggregated'
        }
        daily_forecasts.append(daily_forecast)
    
    return daily_forecasts

def merge_historical_with_current(historical_data: List[Dict], current_data: List[Dict]) -> List[Dict]:
    """Fusionne les données historiques avec les données actuelles"""
    all_data = []
    
    # Ajouter les données historiques
    all_data.extend(historical_data)
    
    # Ajouter les données actuelles avec un marqueur
    for current in current_data:
        current_item = current.copy()
        current_item['date'] = datetime.now().date().isoformat()
        current_item['is_current'] = True
        all_data.append(current_item)
    
    return all_data

def create_master_dataset(files: Dict[str, str]) -> Dict[str, List[Dict]]:
    """Crée le dataset principal en fusionnant toutes les sources"""
    master_data = {}
    
    try:
        # Charger les données
        current_data = []
        forecast_data = []
        air_quality_data = []
        historical_data = []
        
        if files['current']:
            with open(files['current'], 'r') as f:
                current_data = json.load(f)
        
        if files['forecast']:
            with open(files['forecast'], 'r') as f:
                forecast_data = json.load(f)
        
        if files['air_quality']:
            with open(files['air_quality'], 'r') as f:
                air_quality_data = json.load(f)
        
        if files['historical']:
            with open(files['historical'], 'r') as f:
                historical_data = json.load(f)
        
        # Fusionner les données actuelles avec la qualité de l'air
        if current_data and air_quality_data:
            current_enriched = merge_current_and_air_quality(current_data, air_quality_data)
            master_data['current'] = current_enriched
        
        # Agréger les prévisions
        if forecast_data:
            forecast_aggregated = aggregate_forecast_data(forecast_data)
            master_data['forecast'] = forecast_aggregated
        
        # Données historiques
        if historical_data:
            master_data['historical'] = historical_data
        
        # Dataset combiné temporel
        if historical_data and current_data:
            combined_temporal = merge_historical_with_current(historical_data, current_data)
            master_data['combined_temporal'] = combined_temporal
        
        logger.info(f"Dataset principal créé avec {len(master_data)} catégories")
        return master_data
        
    except Exception as e:
        logger.error(f"Erreur lors de la création du dataset principal: {e}")
        return {}

def create_city_comparison_dataset(master_data: Dict[str, List[Dict]]) -> List[Dict]:
    """Crée un dataset optimisé pour la comparaison entre villes"""
    comparison_data = []
    
    if 'current' in master_data:
        for city_data in master_data['current']:
            comparison_item = {
                'city': city_data['city'],
                'country': city_data['country'],
                'lat': city_data['lat'],
                'lon': city_data['lon'],
                'current_temp': city_data['temperature'],
                'current_humidity': city_data['humidity'],
                'current_pressure': city_data['pressure'],
                'current_wind_speed': city_data['wind_speed'],
                'current_weather': city_data['weather_main'],
                'aqi': city_data.get('aqi', None),
                'pm2_5': city_data.get('pm2_5', None),
                'timestamp': city_data['timestamp']
            }
            
            # Ajouter les données historiques moyennes si disponibles
            if 'historical' in master_data:
                city_historical = [h for h in master_data['historical'] if h['city'] == city_data['city']]
                if city_historical:
                    df_hist = pd.DataFrame(city_historical)
                    comparison_item.update({
                        'historical_temp_avg': df_hist['temperature'].mean(),
                        'historical_temp_min': df_hist['temperature'].min(),
                        'historical_temp_max': df_hist['temperature'].max(),
                        'historical_humidity_avg': df_hist['humidity'].mean(),
                        'historical_pressure_avg': df_hist['pressure'].mean(),
                        'temp_deviation': city_data['temperature'] - df_hist['temperature'].mean()
                    })
            
            comparison_data.append(comparison_item)
    
    return comparison_data

def run_merge_process():
    """Exécute le processus de fusion complet"""
    logger.info("Début du processus de fusion des données")
    
    # Trouver les fichiers les plus récents
    files = load_latest_files()
    logger.info(f"Fichiers trouvés: {files}")
    
    # Créer le dataset principal
    master_data = create_master_dataset(files)
    
    # Créer le dataset de comparaison
    comparison_data = create_city_comparison_dataset(master_data)
    
    # Sauvegarder les résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Dataset principal
    master_filename = f"master_weather_data_{timestamp}.json"
    save_data(master_data, master_filename, "processed")
    
    # Dataset de comparaison
    comparison_filename = f"city_comparison_{timestamp}.json"
    save_data(comparison_data, comparison_filename, "processed")
    
    logger.info(f"Fusion terminée. Fichiers créés: {master_filename}, {comparison_filename}")
    
    return {
        'master_file': master_filename,
        'comparison_file': comparison_filename,
        'records_processed': sum(len(v) if isinstance(v, list) else 0 for v in master_data.values())
    }

if __name__ == "__main__":
    run_merge_process()