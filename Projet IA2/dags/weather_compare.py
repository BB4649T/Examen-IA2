import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
import os
import glob
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta
import logging

warnings.filterwarnings('ignore')

# Configuration pour de meilleurs graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WeatherETLAnalyzer:
    def __init__(self):
        """Initialise l'analyseur ETL avec les villes du premier code"""
        self.cities = [
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

        self.climate_profiles = {
            "Paris": {"temp_base": 12, "temp_var": 15, "humidity": 70, "aqi": 2.1},
            "London": {"temp_base": 11, "temp_var": 12, "humidity": 75, "aqi": 1.8},
            "New York": {"temp_base": 13, "temp_var": 20, "humidity": 65, "aqi": 2.3},
            "Tokyo": {"temp_base": 16, "temp_var": 18, "humidity": 68, "aqi": 2.0},
            "Sydney": {"temp_base": 18, "temp_var": 12, "humidity": 65, "aqi": 1.5},
            "Cairo": {"temp_base": 22, "temp_var": 15, "humidity": 45, "aqi": 3.2},
            "Moscow": {"temp_base": 6, "temp_var": 25, "humidity": 72, "aqi": 2.4},
            "Rio de Janeiro": {"temp_base": 24, "temp_var": 8, "humidity": 78, "aqi": 2.7},
            "Mumbai": {"temp_base": 27, "temp_var": 6, "humidity": 80, "aqi": 3.8},
            "Cape Town": {"temp_base": 17, "temp_var": 10, "humidity": 68, "aqi": 1.9}
        }

        self.data_dir = "data"
        self.cities_data = None
        self.setup_directories()
        self.setup_logging()

    def setup_directories(self):
        """Crée les répertoires nécessaires"""
        for subdir in ['raw', 'processed', 'historical', 'reports']:
            os.makedirs(f"{self.data_dir}/{subdir}", exist_ok=True)

    def setup_logging(self):
        """Configure le logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.data_dir}/weather_pipeline.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def calculate_comfort_index(self, temp, humidity, wind_speed):
        """Calcule l'indice de confort comme dans le premier code"""
        heat_index = temp + 0.5 * (humidity - 60) / 100 * (temp - 20)
        wind_chill = temp - 0.5 * wind_speed
        return (heat_index + wind_chill) / 2

    def get_season(self, date, lat):
        """Détermine la saison comme dans le premier code"""
        month = date.month
        if lat >= 0:  # Hémisphère nord
            if month in [12, 1, 2]: return "Hiver"
            elif month in [3, 4, 5]: return "Printemps"
            elif month in [6, 7, 8]: return "Été"
            else: return "Automne"
        else:  # Hémisphère sud
            if month in [12, 1, 2]: return "Été"
            elif month in [3, 4, 5]: return "Automne"
            elif month in [6, 7, 8]: return "Hiver"
            else: return "Printemps"

    def extract_weather_data(self, days_back=30):
        """Extraction des données météo (simule l'ETL extract)"""
        self.logger.info(f"Extraction des données météo pour {len(self.cities)} villes sur {days_back} jours")

        all_data = []
        base_date = datetime.now() - timedelta(days=days_back)

        for i in range(days_back):
            date = base_date + timedelta(days=i)
            for city in self.cities:
                profile = self.climate_profiles[city["name"]]

                # Variation saisonnière basée sur la latitude
                seasonal_factor = np.sin((date.timetuple().tm_yday / 365.25) * 2 * np.pi)
                if city["lat"] < 0:  # Hémisphère sud
                    seasonal_factor = -seasonal_factor

                # Générer les données météo
                temp = profile["temp_base"] + seasonal_factor * profile["temp_var"] + np.random.normal(0, 3)
                humidity = max(20, min(95, profile["humidity"] + np.random.normal(0, 10)))
                pressure = 1013 + np.random.normal(0, 15)
                wind_speed = abs(np.random.gamma(2, 3))
                cloudiness = max(0, min(100, np.random.normal(50, 30)))
                visibility = max(1, min(20, np.random.normal(15, 5)))

                # AQI et composants de pollution
                aqi = max(1, min(5, profile["aqi"] + np.random.normal(0, 0.5)))
                pm25 = max(5, np.random.gamma(profile["aqi"], 15))
                pm10 = pm25 * np.random.uniform(1.2, 2.0)

                # Conditions météorologiques
                weather_conditions = ["Clear", "Clouds", "Rain", "Snow", "Mist", "Thunderstorm"]
                weather_weights = [0.3, 0.3, 0.2, 0.05, 0.1, 0.05]
                weather_main = np.random.choice(weather_conditions, p=weather_weights)

                # Calculer l'indice de confort
                comfort_score = self.calculate_comfort_index(temp, humidity, wind_speed)

                data_point = {
                    'city': city['name'],
                    'country': city['country'],
                    'lat': city['lat'],
                    'lon': city['lon'],
                    'date': date.strftime('%Y-%m-%d'),
                    'timestamp': date.isoformat(),
                    'temperature': round(temp, 1),
                    'feels_like': round(temp + np.random.normal(0, 2), 1),
                    'humidity': round(humidity, 1),
                    'pressure': round(pressure, 1),
                    'wind_speed': round(wind_speed, 1),
                    'wind_direction': np.random.randint(0, 360),
                    'cloudiness': round(cloudiness, 1),
                    'visibility': round(visibility, 1),
                    'weather_main': weather_main,
                    'aqi': round(aqi, 1),
                    'pm2_5': round(pm25, 1),
                    'pm10': round(pm10, 1),
                    'co': round(np.random.gamma(2, 200), 1),
                    'no2': round(np.random.gamma(2, 20), 1),
                    'o3': round(np.random.gamma(2, 50), 1),
                    'comfort_score': round(comfort_score, 2),
                    'season': self.get_season(date, city['lat'])
                }
                all_data.append(data_point)

        # Sauvegarder les données brutes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_filename = f"{self.data_dir}/raw/weather_raw_{timestamp}.json"
        with open(raw_filename, 'w') as f:
            json.dump(all_data, f, indent=2)

        self.logger.info(f"Données extraites et sauvegardées: {raw_filename}")
        return all_data

    def transform_data(self, raw_data):
        """Transformation des données (simule l'ETL transform)"""
        self.logger.info("Transformation des données en cours...")

        # Convertir en DataFrame
        df = pd.DataFrame(raw_data)
        df['date'] = pd.to_datetime(df['date'])

        # Ajout de métriques dérivées
        df['temp_category'] = pd.cut(df['temperature'],
                                   bins=[-np.inf, 0, 15, 25, np.inf],
                                   labels=['Froid', 'Frais', 'Tempéré', 'Chaud'])

        df['humidity_category'] = pd.cut(df['humidity'],
                                       bins=[0, 30, 60, 80, 100],
                                       labels=['Sec', 'Modéré', 'Humide', 'Très humide'])

        # Catégorisation AQI
        df['aqi_category'] = pd.cut(df['aqi'],
                                  bins=[0, 1, 2, 3, 4, 5],
                                  labels=['Excellent', 'Bon', 'Modéré', 'Mauvais', 'Très mauvais'])

        # Calcul des moyennes mobiles
        df = df.sort_values(['city', 'date'])
        df['temp_ma_7'] = df.groupby('city')['temperature'].transform(lambda x: x.rolling(7, min_periods=1).mean())
        df['humidity_ma_7'] = df.groupby('city')['humidity'].transform(lambda x: x.rolling(7, min_periods=1).mean())

        self.cities_data = df

        # Sauvegarder les données transformées
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_filename = f"{self.data_dir}/processed/weather_processed_{timestamp}.csv"
        df.to_csv(processed_filename, index=False)

        self.logger.info(f"Données transformées et sauvegardées: {processed_filename}")
        return df

    def basic_stats_analysis(self):
        """Analyse statistique de base du premier code"""
        self.logger.info("Génération des statistiques de base...")

        stats = {
            'period': f"{self.cities_data['date'].min()} à {self.cities_data['date'].max()}",
            'total_observations': len(self.cities_data),
            'cities_count': self.cities_data['city'].nunique(),
            'countries_count': self.cities_data['country'].nunique(),
            'descriptive_stats': {}
        }

        numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'aqi', 'pm2_5', 'comfort_score']
        for col in numeric_cols:
            stats['descriptive_stats'][col] = {
                'mean': float(self.cities_data[col].mean()),
                'std': float(self.cities_data[col].std()),
                'min': float(self.cities_data[col].min()),
                'max': float(self.cities_data[col].max())
            }

        return stats

    def city_comparison_analysis(self):
        """Comparaison entre villes du premier code"""
        self.logger.info("Analyse comparative des villes...")

        city_means = self.cities_data.groupby('city').agg({
            'temperature': 'mean',
            'humidity': 'mean',
            'pressure': 'mean',
            'wind_speed': 'mean',
            'aqi': 'mean',
            'pm2_5': 'mean',
            'comfort_score': 'mean'
        }).round(2)

        # Classements comme dans le premier code
        rankings = {
            'hottest_cities': city_means.sort_values('temperature', ascending=False)['temperature'].head().to_dict(),
            'windiest_cities': city_means.sort_values('wind_speed', ascending=False)['wind_speed'].head().to_dict(),
            'best_air_quality': city_means.sort_values('aqi', ascending=True)['aqi'].head().to_dict(),
            'most_comfortable': city_means.sort_values('comfort_score', ascending=False)['comfort_score'].head().to_dict(),
            'most_stable': self.cities_data.groupby('city')['temperature'].std().sort_values().head().to_dict()
        }

        return {
            'city_means': city_means.to_dict(),
            'rankings': rankings
        }

    def weather_patterns_analysis(self):
        """Analyse des patterns météorologiques du premier code"""
        self.logger.info("Analyse des patterns météorologiques...")

        # Distribution des conditions météo
        weather_dist = self.cities_data['weather_main'].value_counts().to_dict()

        # Analyse saisonnière
        seasonal_temp = self.cities_data.groupby(['season', 'city'])['temperature'].mean().unstack()

        return {
            'weather_distribution': weather_dist,
            'seasonal_temperatures': seasonal_temp.to_dict() if not seasonal_temp.empty else {}
        }

    def air_quality_analysis(self):
        """Analyse de la qualité de l'air du premier code"""
        self.logger.info("Analyse de la qualité de l'air...")

        # Distribution par catégorie AQI
        aqi_dist = self.cities_data['aqi_category'].value_counts().to_dict()

        # Qualité d'air par ville
        city_air_quality = self.cities_data.groupby('city').agg({
            'aqi': 'mean',
            'pm2_5': 'mean',
            'pm10': 'mean'
        }).round(2)

        return {
            'aqi_distribution': aqi_dist,
            'city_air_quality': city_air_quality.to_dict()
        }

    def generate_insights(self):
        """Génère les insights principaux du premier code"""
        self.logger.info("Génération des insights...")

        # Ville la plus chaude/froide
        avg_temp_by_city = self.cities_data.groupby('city')['temperature'].mean()
        hottest_city = avg_temp_by_city.idxmax()
        coldest_city = avg_temp_by_city.idxmin()

        # Qualité d'air
        avg_aqi_by_city = self.cities_data.groupby('city')['aqi'].mean()
        best_air = avg_aqi_by_city.idxmin()
        worst_air = avg_aqi_by_city.idxmax()

        # Confort climatique
        avg_comfort_by_city = self.cities_data.groupby('city')['comfort_score'].mean()
        most_comfortable = avg_comfort_by_city.idxmax()

        # Stabilité météorologique
        temp_stability = self.cities_data.groupby('city')['temperature'].std()
        most_stable = temp_stability.idxmin()

        # Condition météo dominante
        dominant_weather = self.cities_data['weather_main'].mode()[0]

        insights = {
            'hottest_city': {'city': hottest_city, 'temperature': float(avg_temp_by_city[hottest_city])},
            'coldest_city': {'city': coldest_city, 'temperature': float(avg_temp_by_city[coldest_city])},
            'best_air_quality': {'city': best_air, 'aqi': float(avg_aqi_by_city[best_air])},
            'worst_air_quality': {'city': worst_air, 'aqi': float(avg_aqi_by_city[worst_air])},
            'most_comfortable': {'city': most_comfortable, 'score': float(avg_comfort_by_city[most_comfortable])},
            'most_stable': {'city': most_stable, 'std': float(temp_stability[most_stable])},
            'dominant_weather': dominant_weather
        }

        return insights

    def create_visualizations(self):
        """Crée les visualisations du premier code"""
        self.logger.info("Création des visualisations...")

        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Analyse Exploratoire des Données Météorologiques ETL', fontsize=16, fontweight='bold')

        # 1. Distribution des températures par ville
        self.cities_data.boxplot(column='temperature', by='city', ax=axes[0,0])
        axes[0,0].set_title('Distribution des Températures par Ville')
        axes[0,0].set_xlabel('Ville')
        axes[0,0].set_ylabel('Température (°C)')
        axes[0,0].tick_params(axis='x', rotation=45)

        # 2. Qualité de l'air par ville
        city_aqi = self.cities_data.groupby('city')['aqi'].mean().sort_values()
        city_aqi.plot(kind='bar', ax=axes[0,1], color='orange')
        axes[0,1].set_title('Qualité de l\'Air Moyenne par Ville (AQI)')
        axes[0,1].set_xlabel('Ville')
        axes[0,1].set_ylabel('AQI')
        axes[0,1].tick_params(axis='x', rotation=45)

        # 3. Corrélation température vs humidité
        scatter = axes[1,0].scatter(self.cities_data['temperature'], self.cities_data['humidity'],
                                   c=self.cities_data['comfort_score'], cmap='RdYlBu_r', alpha=0.6)
        axes[1,0].set_xlabel('Température (°C)')
        axes[1,0].set_ylabel('Humidité (%)')
        axes[1,0].set_title('Température vs Humidité (couleur = confort)')
        plt.colorbar(scatter, ax=axes[1,0])

        # 4. Distribution des conditions météo
        weather_counts = self.cities_data['weather_main'].value_counts()
        axes[1,1].pie(weather_counts.values, labels=weather_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('Distribution des Conditions Météorologiques')

        # 5. Evolution temporelle des températures
        daily_temp = self.cities_data.groupby(['date', 'city'])['temperature'].mean().unstack()
        for city in daily_temp.columns[:5]:  # Afficher 5 villes
            axes[2,0].plot(daily_temp.index, daily_temp[city],
                          label=city, marker='o', markersize=3)
        axes[2,0].set_title('Évolution des Températures dans le Temps')
        axes[2,0].set_xlabel('Date')
        axes[2,0].set_ylabel('Température (°C)')
        axes[2,0].legend()
        axes[2,0].tick_params(axis='x', rotation=45)

        # 6. Heatmap des corrélations
        numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'aqi', 'pm2_5', 'comfort_score']
        corr_matrix = self.cities_data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[2,1])
        axes[2,1].set_title('Matrice de Corrélation')

        plt.tight_layout()

        # Sauvegarder les visualisations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_filename = f"{self.data_dir}/reports/weather_visualizations_{timestamp}.png"
        plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Visualisations sauvegardées: {viz_filename}")
        return viz_filename

    def run_complete_analysis(self):
        """Exécute l'analyse complète intégrant l'ETL et l'EDA"""
        self.logger.info("🌍 DÉBUT DE L'ANALYSE MÉTÉOROLOGIQUE ETL COMPLÈTE")

        # 1. Extraction des données
        raw_data = self.extract_weather_data()

        # 2. Transformation des données
        transformed_data = self.transform_data(raw_data)

        # 3. Analyses comme dans le premier code
        basic_stats = self.basic_stats_analysis()
        city_comparison = self.city_comparison_analysis()
        weather_patterns = self.weather_patterns_analysis()
        air_quality = self.air_quality_analysis()
        insights = self.generate_insights()

        # 4. Visualisations
        viz_file = self.create_visualizations()

        # 5. Compilation du rapport final
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'basic_statistics': basic_stats,
            'city_comparison': city_comparison,
            'weather_patterns': weather_patterns,
            'air_quality': air_quality,
            'insights': insights,
            'visualization_file': viz_file,
            'data_quality': {
                'total_records': len(self.cities_data),
                'missing_values': self.cities_data.isnull().sum().to_dict(),
                'cities_analyzed': list(self.cities_data['city'].unique())
            }
        }

        # Sauvegarder le rapport final
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{self.data_dir}/reports/weather_complete_analysis_{timestamp}.json"
        with open(report_filename, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)

        # Sauvegarder aussi le CSV final
        csv_filename = f"{self.data_dir}/reports/weather_final_data_{timestamp}.csv"
        self.cities_data.to_csv(csv_filename, index=False)

        self.logger.info(f"✅ Analyse complète terminée!")
        self.logger.info(f"📊 Rapport: {report_filename}")
        self.logger.info(f"📈 Visualisations: {viz_file}")
        self.logger.info(f"💾 Données CSV: {csv_filename}")

        return final_report

# Fonctions pour l'intégration Airflow
def run_current_extraction():
    """Fonction pour l'extraction des données actuelles"""
    analyzer = WeatherETLAnalyzer()
    return analyzer.extract_weather_data(days_back=1)

def run_historical_extraction(days_back=7):
    """Fonction pour l'extraction des données historiques"""
    analyzer = WeatherETLAnalyzer()
    return analyzer.extract_weather_data(days_back=days_back)

def run_merge_process():
    """Fonction pour fusionner les données"""
    analyzer = WeatherETLAnalyzer()
    # Charger les données les plus récentes
    raw_files = glob.glob(f"{analyzer.data_dir}/raw/weather_raw_*.json")
    if raw_files:
        latest_file = max(raw_files, key=os.path.getctime)
        with open(latest_file, 'r') as f:
            raw_data = json.load(f)
        return analyzer.transform_data(raw_data)
    return None

def run_transformation_analysis():
    """Fonction pour l'analyse complète"""
    analyzer = WeatherETLAnalyzer()
    return analyzer.run_complete_analysis()

def validate_data_quality():
    """Validation de la qualité des données"""
    analyzer = WeatherETLAnalyzer()

    # Vérifier les fichiers récents
    processed_files = glob.glob(f"{analyzer.data_dir}/processed/weather_processed_*.csv")
    if not processed_files:
        raise ValueError("Aucun fichier de données traitées trouvé")

    # Charger et valider le dernier fichier
    latest_file = max(processed_files, key=os.path.getctime)
    df = pd.read_csv(latest_file)

    if len(df) < 100:  # Au moins 100 observations
        raise ValueError(f"Données insuffisantes: seulement {len(df)} observations")

    # Vérifier les champs essentiels
    required_fields = ['city', 'temperature', 'humidity', 'pressure', 'aqi', 'comfort_score']
    missing_fields = [field for field in required_fields if field not in df.columns]
    if missing_fields:
        raise ValueError(f"Champs manquants: {missing_fields}")

    analyzer.logger.info(f"Validation réussie: {len(df)} observations avec données complètes")
    return True

def generate_summary_report():
    """Génère un rapport de synthèse"""
    analyzer = WeatherETLAnalyzer()

    # Charger les résultats d'analyse les plus récents
    report_files = glob.glob(f"{analyzer.data_dir}/reports/weather_complete_analysis_*.json")
    if not report_files:
        analyzer.logger.warning("Aucun fichier d'analyse trouvé")
        return

    latest_analysis = max(report_files, key=os.path.getctime)
    with open(latest_analysis, 'r') as f:
        analysis_data = json.load(f)

    # Créer le rapport de synthèse
    summary = {
        'date': datetime.now().isoformat(),
        'cities_analyzed': len(analysis_data['insights']),
        'hottest_city': analysis_data['insights']['hottest_city'],
        'coldest_city': analysis_data['insights']['coldest_city'],
        'best_air_quality': analysis_data['insights']['best_air_quality'],
        'most_comfortable': analysis_data['insights']['most_comfortable'],
        'summary': f"Analyse de {len(analyzer.cities)} villes terminée avec succès"
    }

    # Sauvegarder le rapport de synthèse
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"{analyzer.data_dir}/reports/daily_weather_summary_{timestamp}.json"

    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)

    analyzer.logger.info(f"Rapport de synthèse généré: {summary_filename}")
    return summary

def cleanup_old_files():
    """Nettoie les fichiers anciens"""
    analyzer = WeatherETLAnalyzer()
    cutoff_date = datetime.now() - timedelta(days=30)

    # Nettoyer les fichiers anciens
    for subfolder in ['raw', 'processed', 'reports']:
        folder_path = f"{analyzer.data_dir}/{subfolder}"
        if os.path.exists(folder_path):
            for filepath in glob.glob(f"{folder_path}/*"):
                if os.path.isfile(filepath):
                    file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                    if file_time < cutoff_date:
                        os.remove(filepath)
                        analyzer.logger.info(f"Fichier supprimé: {filepath}")

def create_data_directory():
    """Crée le répertoire de données"""
    analyzer = WeatherETLAnalyzer()
    analyzer.setup_directories()

def setup_logging():
    """Configure le logging"""
    return logging.getLogger(__name__)

# Configuration du DAG Airflow
default_args = {
    'owner': 'weather-team',
    'depends_on_past': False,
    'start_date': datetime.now() - timedelta(days=1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

dag = DAG(
    'weather_comparison_pipeline_enhanced',
    default_args=default_args,
    description='Pipeline de comparaison climatique avec analyse EDA complète',
    schedule='0 6 * * *',  # <- Remplacer schedule_interval par schedule
    max_active_runs=1,
    tags=['weather', 'climate', 'comparison', 'cities', 'eda']
)

# Définition des tâches Airflow
task_init = PythonOperator(
    task_id='initialize_environment',
    python_callable=create_data_directory,
    dag=dag
)

task_extract_current = PythonOperator(
    task_id='extract_current_weather',
    python_callable=run_current_extraction,
    dag=dag
)

task_extract_historical = PythonOperator(
    task_id='extract_historical_weather',
    python_callable=lambda: run_historical_extraction(days_back=7),
    dag=dag
)

task_validate_data = PythonOperator(
    task_id='validate_data_quality',python_callable=validate_data_quality,
    dag=dag
)

task_merge_data = PythonOperator(
    task_id='merge_weather_data',
    python_callable=run_merge_process,
    dag=dag
)

task_transform_analyze = PythonOperator(
    task_id='transform_and_analyze',
    python_callable=run_transformation_analysis,
    dag=dag
)

task_generate_report = PythonOperator(
    task_id='generate_summary_report',
    python_callable=generate_summary_report,
    dag=dag
)

task_cleanup = PythonOperator(
    task_id='cleanup_old_files',
    python_callable=cleanup_old_files,
    dag=dag
)

# Tâche pour générer le dashboard
task_dashboard = BashOperator(
    task_id='update_dashboard',
    bash_command='echo "Dashboard mis à jour - analyse EDA complète terminée"',
    dag=dag
)

# Définition des dépendances
task_init >> [task_extract_current, task_extract_historical]
[task_extract_current, task_extract_historical] >> task_validate_data
task_validate_data >> task_merge_data
task_merge_data >> task_transform_analyze
task_transform_analyze >> task_generate_report
task_generate_report >> task_dashboard
task_dashboard >> task_cleanup

# Configuration des alertes
def task_fail_alert(context):
    """Alerte en cas d'échec d'une tâche"""
    logger = setup_logging()
    task_id = context['task_instance'].task_id
    dag_id = context['dag'].dag_id
    execution_date = context['execution_date']

    logger.error(f"Échec de la tâche {task_id} dans le DAG {dag_id} le {execution_date}")

    # Ici vous pouvez ajouter des notifications (email, Slack, etc.)

# Appliquer les alertes aux tâches critiques
for task in [task_extract_current, task_merge_data, task_transform_analyze]:
    task.on_failure_callback = task_fail_alert

# Fonction principale pour exécution standalone
if __name__ == "__main__":
    print("🌍 LANCEMENT DE L'ANALYSE MÉTÉOROLOGIQUE ETL COMPLÈTE")
    print("=" * 70)

    # Créer et exécuter l'analyse complète
    analyzer = WeatherETLAnalyzer()
    results = analyzer.run_complete_analysis()

    print("\n📊 RÉSULTATS DE L'ANALYSE:")
    print(f"✅ Villes analysées: {len(results['data_quality']['cities_analyzed'])}")
    print(f"📈 Total d'observations: {results['data_quality']['total_records']}")
    print(f"🌡️ Ville la plus chaude: {results['insights']['hottest_city']['city']} ({results['insights']['hottest_city']['temperature']:.1f}°C)")
    print(f"❄️ Ville la plus froide: {results['insights']['coldest_city']['city']} ({results['insights']['coldest_city']['temperature']:.1f}°C)")
    print(f"🌿 Meilleure qualité d'air: {results['insights']['best_air_quality']['city']} (AQI: {results['insights']['best_air_quality']['aqi']:.1f})")
    print(f"😊 Ville la plus confortable: {results['insights']['most_comfortable']['city']} (Score: {results['insights']['most_comfortable']['score']:.2f})")
    print(f"📊 Ville la plus stable: {results['insights']['most_stable']['city']} (Écart-type: {results['insights']['most_stable']['std']:.1f}°C)")
    print(f"☁️ Condition météo dominante: {results['insights']['dominant_weather']}")

    print(f"\n💾 Fichiers générés:")
    print(f"- Rapport JSON: {results['visualization_file'].replace('.png', '.json').replace('visualizations', 'complete_analysis')}")
    print(f"- Visualisations: {results['visualization_file']}")
    print(f"- Données CSV: {results['visualization_file'].replace('visualizations', 'final_data').replace('.png', '.csv')}")

    print("\n🎯 VILLES ANALYSÉES:")
    for city in results['data_quality']['cities_analyzed']:
        print(f"  • {city}")

    print("\n✅ Analyse ETL complète terminée avec succès!")
    print("📈 Toutes les fonctionnalités du premier code ont été intégrées")
    print("🔄 Structure ETL maintenue pour l'intégration Airflow")
