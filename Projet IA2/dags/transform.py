import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import json
import time
from utils import (
    DATA_DIR, save_data, load_data, setup_logging, 
    calculate_comfort_index, get_season, CITIES
)

logger = setup_logging()

def load_latest_merged_data() -> Dict:
    """Charge les données fusionnées les plus récentes"""
    import glob
    import os
    
    files = glob.glob(f"{DATA_DIR}/processed/master_weather_data_*.json")
    if not files:
        logger.error("Aucun fichier de données fusionnées trouvé")
        return {}
    
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        return json.load(f)

def calculate_temperature_stability(city_data: List[Dict]) -> Dict:
    """Calcule les indicateurs de stabilité thermique"""
    if not city_data:
        return {}
    
    temps = [d['temperature'] for d in city_data]
    
    stability_metrics = {
        'temp_variance': np.var(temps),
        'temp_std': np.std(temps),
        'temp_range': max(temps) - min(temps),
        'temp_stability_score': 1 / (1 + np.std(temps))  # Score entre 0 et 1
    }
    
    return stability_metrics

def calculate_climate_extremes(city_data: List[Dict]) -> Dict:
    """Calcule les indices d'extrêmes climatiques"""
    if not city_data:
        return {}
    
    df = pd.DataFrame(city_data)
    
    extremes = {
        'temp_extreme_hot_days': len(df[df['temperature'] > df['temperature'].quantile(0.9)]),
        'temp_extreme_cold_days': len(df[df['temperature'] < df['temperature'].quantile(0.1)]),
        'high_humidity_days': len(df[df['humidity'] > 80]),
        'low_humidity_days': len(df[df['humidity'] < 30]),
        'high_wind_days': len(df[df['wind_speed'] > df['wind_speed'].quantile(0.9)]),
        'extreme_weather_score': 0
    }
    
    # Score d'extrêmes (plus élevé = plus d'extrêmes)
    total_days = len(df)
    if total_days > 0:
        extreme_ratio = (extremes['temp_extreme_hot_days'] + 
                        extremes['temp_extreme_cold_days'] + 
                        extremes['high_humidity_days'] + 
                        extremes['low_humidity_days']) / (total_days * 4)
        extremes['extreme_weather_score'] = extreme_ratio
    
    return extremes

def calculate_comfort_metrics(city_data: List[Dict]) -> Dict:
    """Calcule les métriques de confort climatique"""
    if not city_data:
        return {}
    
    comfort_scores = []
    comfortable_days = 0
    
    for data in city_data:
        comfort_score = calculate_comfort_index(
            data['temperature'], 
            data['humidity'], 
            data['wind_speed']
        )
        comfort_scores.append(comfort_score)
        
        # Jour confortable : 18-25°C, humidité 40-60%, vent < 20 km/h
        if (18 <= data['temperature'] <= 25 and 
            40 <= data['humidity'] <= 60 and 
            data['wind_speed'] < 20):
            comfortable_days += 1
    
    comfort_metrics = {
        'avg_comfort_score': np.mean(comfort_scores),
        'comfort_score_std': np.std(comfort_scores),
        'comfortable_days_ratio': comfortable_days / len(city_data) if city_data else 0,
        'comfort_consistency': 1 / (1 + np.std(comfort_scores))
    }
    
    return comfort_metrics

def calculate_seasonal_patterns(city_data: List[Dict]) -> Dict:
    """Analyse les patterns saisonniers"""
    if not city_data:
        return {}
    
    # Grouper par saison
    seasonal_data = {}
    for data in city_data:
        if 'season' in data:
            season = data['season']
            if season not in seasonal_data:
                seasonal_data[season] = []
            seasonal_data[season].append(data)
    
    seasonal_metrics = {}
    
    for season, season_data in seasonal_data.items():
        if season_data:
            temps = [d['temperature'] for d in season_data]
            seasonal_metrics[f'{season.lower()}_temp_avg'] = np.mean(temps)
            seasonal_metrics[f'{season.lower()}_temp_range'] = max(temps) - min(temps)
            seasonal_metrics[f'{season.lower()}_humidity_avg'] = np.mean([d['humidity'] for d in season_data])
    
    # Calculer la variabilité saisonnière
    if len(seasonal_metrics) >= 4:  # Si on a les 4 saisons
        seasonal_temps = [seasonal_metrics[f'{s.lower()}_temp_avg'] 
                         for s in ['Hiver', 'Printemps', 'Été', 'Automne'] 
                         if f'{s.lower()}_temp_avg' in seasonal_metrics]
        
        seasonal_metrics['seasonal_temperature_range'] = max(seasonal_temps) - min(seasonal_temps)
        seasonal_metrics['seasonal_variability'] = np.std(seasonal_temps)
    
    return seasonal_metrics

def calculate_weather_diversity(city_data: List[Dict]) -> Dict:
    """Calcule la diversité des conditions météorologiques"""
    if not city_data:
        return {}
    
    weather_types = [d['weather_main'] for d in city_data]
    unique_weather = set(weather_types)
    
    # Compter les occurrences
    weather_counts = {}
    for weather in weather_types:
        weather_counts[weather] = weather_counts.get(weather, 0) + 1
    
    # Calculer l'indice de diversité de Shannon
    total = len(weather_types)
    shannon_diversity = 0
    for count in weather_counts.values():
        p = count / total
        shannon_diversity -= p * np.log(p)
    
    diversity_metrics = {
        'weather_types_count': len(unique_weather),
        'weather_diversity_index': shannon_diversity,
        'dominant_weather': max(weather_counts, key=weather_counts.get),
        'weather_distribution': weather_counts
    }
    
    return diversity_metrics

def calculate_air_quality_score(city_data: List[Dict]) -> Dict:
    """Calcule le score de qualité de l'air"""
    air_quality_data = [d for d in city_data if 'aqi' in d and d['aqi'] is not None]
    
    if not air_quality_data:
        return {'air_quality_score': None, 'air_quality_category': 'Unknown'}
    
    avg_aqi = np.mean([d['aqi'] for d in air_quality_data])
    
    # Catégorisation AQI
    if avg_aqi <= 1:
        category = 'Excellent'
    elif avg_aqi <= 2:
        category = 'Bon'
    elif avg_aqi <= 3:
        category = 'Modéré'
    elif avg_aqi <= 4:
        category = 'Mauvais'
    else:
        category = 'Très mauvais'
    
    air_metrics = {
        'air_quality_score': avg_aqi,
        'air_quality_category': category,
        'pm2_5_avg': np.mean([d.get('pm2_5', 0) for d in air_quality_data]),
        'pm10_avg': np.mean([d.get('pm10', 0) for d in air_quality_data])
    }
    
    return air_metrics

def calculate_precipitation_patterns(city_data: List[Dict]) -> Dict:
    """Analyse les patterns de précipitations"""
    precipitation_data = []
    
    for data in city_data:
        # Chercher les données de pluie dans différents champs
        rain = 0
        if 'rain_3h' in data:
            rain += data['rain_3h']
        if 'rain_total' in data:
            rain += data['rain_total']
        
        precipitation_data.append({
            'date': data.get('date', data.get('timestamp', '')),
            'rain': rain,
            'humidity': data['humidity'],
            'cloudiness': data.get('cloudiness', 0)
        })
    
    if not precipitation_data:
        return {}
    
    rainy_days = len([d for d in precipitation_data if d['rain'] > 0])
    total_rain = sum(d['rain'] for d in precipitation_data)
    
    precipitation_metrics = {
        'rainy_days_count': rainy_days,
        'rainy_days_ratio': rainy_days / len(precipitation_data),
        'total_precipitation': total_rain,
        'avg_daily_precipitation': total_rain / len(precipitation_data),
        'avg_humidity': np.mean([d['humidity'] for d in precipitation_data]),
        'avg_cloudiness': np.mean([d['cloudiness'] for d in precipitation_data])
    }
    
    return precipitation_metrics

def create_city_ranking(cities_metrics: Dict[str, Dict]) -> Dict:
    """Crée un classement des villes selon différents critères"""
    
    rankings = {
        'most_stable': [],
        'most_extreme': [],
        'most_comfortable': [],
        'best_air_quality': [],
        'most_diverse_weather': []
    }
    
    # Classement par stabilité (score de stabilité décroissant)
    stable_cities = [(city, metrics.get('temp_stability_score', 0)) 
                     for city, metrics in cities_metrics.items()]
    rankings['most_stable'] = sorted(stable_cities, key=lambda x: x[1], reverse=True)
    
    # Classement par extrêmes (score d'extrêmes croissant)
    extreme_cities = [(city, metrics.get('extreme_weather_score', 0)) 
                      for city, metrics in cities_metrics.items()]
    rankings['most_extreme'] = sorted(extreme_cities, key=lambda x: x[1], reverse=True)
    
    # Classement par confort
    comfort_cities = [(city, metrics.get('avg_comfort_score', 0)) 
                      for city, metrics in cities_metrics.items()]
    rankings['most_comfortable'] = sorted(comfort_cities, key=lambda x: x[1], reverse=True)
    
    # Classement par qualité de l'air (AQI décroissant = meilleure qualité)
    air_cities = [(city, metrics.get('air_quality_score', 5)) 
                  for city, metrics in cities_metrics.items() 
                  if metrics.get('air_quality_score') is not None]
    rankings['best_air_quality'] = sorted(air_cities, key=lambda x: x[1])
    
    # Classement par diversité météorologique
    diversity_cities = [(city, metrics.get('weather_diversity_index', 0)) 
                        for city, metrics in cities_metrics.items()]
    rankings['most_diverse_weather'] = sorted(diversity_cities, key=lambda x: x[1], reverse=True)
    
    return rankings

def generate_city_insights(city_name: str, city_metrics: Dict) -> Dict:
    """Génère des insights pour une ville spécifique"""
    insights = {
        'city': city_name,
        'overall_score': 0,
        'strengths': [],
        'weaknesses': [],
        'climate_type': 'Unknown',
        'recommendations': []
    }
    
    # Déterminer le type de climat
    temp_range = city_metrics.get('seasonal_temperature_range', 0)
    humidity_avg = city_metrics.get('avg_humidity', 50)
    precipitation_ratio = city_metrics.get('rainy_days_ratio', 0)
    
    if temp_range < 15:
        if humidity_avg > 70:
            insights['climate_type'] = 'Océanique'
        else:
            insights['climate_type'] = 'Méditerranéen'
    elif temp_range > 30:
        if precipitation_ratio < 0.2:
            insights['climate_type'] = 'Continental sec'
        else:
            insights['climate_type'] = 'Continental humide'
    else:
        insights['climate_type'] = 'Tempéré'
    
    # Identifier les forces
    if city_metrics.get('temp_stability_score', 0) > 0.7:
        insights['strengths'].append('Température très stable')
    
    if city_metrics.get('comfortable_days_ratio', 0) > 0.6:
        insights['strengths'].append('Nombreux jours confortables')
    
    if city_metrics.get('air_quality_score', 5) < 2:
        insights['strengths'].append('Excellente qualité de l\'air')
    
    # Identifier les faiblesses
    if city_metrics.get('extreme_weather_score', 0) > 0.3:
        insights['weaknesses'].append('Conditions météo extrêmes fréquentes')
    
    if city_metrics.get('air_quality_score', 0) > 3:
        insights['weaknesses'].append('Qualité de l\'air préoccupante')
    
    if city_metrics.get('rainy_days_ratio', 0) > 0.4:
        insights['weaknesses'].append('Précipitations fréquentes')
    
    # Calcul du score global
    stability_score = city_metrics.get('temp_stability_score', 0) * 0.3
    comfort_score = city_metrics.get('comfortable_days_ratio', 0) * 0.3
    air_score = (5 - city_metrics.get('air_quality_score', 5)) / 5 * 0.2
    diversity_score = min(city_metrics.get('weather_diversity_index', 0) / 2, 1) * 0.2
    
    insights['overall_score'] = stability_score + comfort_score + air_score + diversity_score
    
    return insights

def run_transformation_analysis():
    """Exécute l'analyse complète de transformation"""
    logger.info("Début de l'analyse de transformation des données")
    
    # Charger les données fusionnées
    master_data = load_latest_merged_data()
    
    if not master_data:
        logger.error("Aucune donnée à analyser")
        return {}
    
    # Analyser chaque ville
    cities_metrics = {}
    cities_insights = {}
    
    for city_info in CITIES:
        city_name = city_info['name']
        logger.info(f"Analyse de {city_name}")
        
        # Collecter toutes les données de la ville
        city_data = []
        
        # Données actuelles
        if 'current' in master_data:
            city_current = [d for d in master_data['current'] if d['city'] == city_name]
            city_data.extend(city_current)
        
        # Données historiques
        if 'historical' in master_data:
            city_historical = [d for d in master_data['historical'] if d['city'] == city_name]
            city_data.extend(city_historical)
        
        # Données de prévision
        if 'forecast' in master_data:
            city_forecast = [d for d in master_data['forecast'] if d['city'] == city_name]
            city_data.extend(city_forecast)
        
        if not city_data:
            logger.warning(f"Aucune donnée trouvée pour {city_name}")
            continue
        
        # Calculer les métriques
        metrics = {}
        
        # Stabilité thermique
        metrics.update(calculate_temperature_stability(city_data))
        
        # Extrêmes climatiques
        metrics.update(calculate_climate_extremes(city_data))
        
        # Confort climatique
        metrics.update(calculate_comfort_metrics(city_data))
        
        # Patterns saisonniers
        metrics.update(calculate_seasonal_patterns(city_data))
        
        # Diversité météorologique
        metrics.update(calculate_weather_diversity(city_data))
        
        # Qualité de l'air
        metrics.update(calculate_air_quality_score(city_data))
        
        # Précipitations
        metrics.update(calculate_precipitation_patterns(city_data))
        
        cities_metrics[city_name] = metrics
        
        # Générer les insights
        cities_insights[city_name] = generate_city_insights(city_name, metrics)
    
    # Créer les classements
    rankings = create_city_ranking(cities_metrics)
    
    # Préparer les résultats finaux
    analysis_results = {
        'cities_metrics': cities_metrics,
        'cities_insights': cities_insights,
        'rankings': rankings,
        'analysis_timestamp': datetime.now().isoformat(),
        'cities_analyzed': len(cities_metrics)
    }
    
    # Sauvegarder les résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Métriques détaillées
    metrics_filename = f"cities_metrics_{timestamp}.json"
    save_data(cities_metrics, metrics_filename, "processed")
    
    # Insights et classements
    insights_filename = f"climate_analysis_{timestamp}.json"
    save_data(analysis_results, insights_filename, "processed")
    
    logger.info(f"Analyse terminée. {len(cities_metrics)} villes analysées")
    
    return {
        'metrics_file': metrics_filename,
        'analysis_file': insights_filename,
        'cities_analyzed': len(cities_metrics)
    }

if __name__ == "__main__":
    run_transformation_analysis()