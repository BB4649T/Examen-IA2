import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Météo",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🌦️ Dashboard Météo - Pipeline ETL Automatisé")

# Sidebar pour les filtres
st.sidebar.header("Filtres")

@st.cache_data
def load_data():
    """Charge les données météo depuis le fichier CSV"""
    try:
        df = pd.read_csv("weather_data_sample.csv")
        df['date'] = pd.to_datetime(df['date'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        st.error("⚠️ Fichier weather_data_sample.csv non trouvé")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

# Chargement des données
df = load_data()

if df is not None:
    # Filtres dans la sidebar
    cities = st.sidebar.multiselect(
        "Sélectionner les villes",
        options=df['city'].unique(),
        default=df['city'].unique()[:3]  # Sélection des 3 premières villes par défaut
    )

    date_range = st.sidebar.date_input(
        "Période",
        value=(df['date'].min(), df['date'].max()),
        min_value=df['date'].min(),
        max_value=df['date'].max()
    )

    # Filtrage des données
    df_filtered = df[
        (df['city'].isin(cities)) &
        (df['date'] >= pd.to_datetime(date_range[0])) &
        (df['date'] <= pd.to_datetime(date_range[1]))
    ]

    # Métriques principales
    st.header("📊 Indicateurs Clés")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_temp = df_filtered['temperature'].mean()
        st.metric(
            label="Température Moyenne",
            value=f"{avg_temp:.1f}°C",
            delta=f"{avg_temp - df['temperature'].mean():.1f}°C"
        )

    with col2:
        avg_humidity = df_filtered['humidity'].mean()
        st.metric(
            label="Humidité Moyenne",
            value=f"{avg_humidity:.1f}%",
            delta=f"{avg_humidity - df['humidity'].mean():.1f}%"
        )

    with col3:
        avg_aqi = df_filtered['aqi'].mean()
        st.metric(
            label="Indice Qualité Air",
            value=f"{avg_aqi:.1f}",
            delta=f"{avg_aqi - df['aqi'].mean():.1f}"
        )

    with col4:
        avg_comfort = df_filtered['comfort_score'].mean()
        st.metric(
            label="Score Confort",
            value=f"{avg_comfort:.1f}/20",
            delta=f"{avg_comfort - df['comfort_score'].mean():.1f}"
        )

    # Graphiques principaux
    st.header("📈 Évolution Temporelle")

    col1, col2 = st.columns(2)

    with col1:
        # Graphique température
        fig_temp = px.line(
            df_filtered,
            x='date',
            y='temperature',
            color='city',
            title="Évolution de la Température",
            labels={'temperature': 'Température (°C)', 'date': 'Date'}
        )
        fig_temp.update_layout(height=400)
        st.plotly_chart(fig_temp, use_container_width=True)

    with col2:
        # Graphique humidité
        fig_humidity = px.line(
            df_filtered,
            x='date',
            y='humidity',
            color='city',
            title="Évolution de l'Humidité",
            labels={'humidity': 'Humidité (%)', 'date': 'Date'}
        )
        fig_humidity.update_layout(height=400)
        st.plotly_chart(fig_humidity, use_container_width=True)

    # Graphiques de qualité de l'air
    st.header("🏭 Qualité de l'Air")

    col1, col2 = st.columns(2)

    with col1:
        # Graphique AQI
        fig_aqi = px.bar(
            df_filtered.groupby('city')['aqi'].mean().reset_index(),
            x='city',
            y='aqi',
            title="Indice Qualité Air Moyen par Ville",
            labels={'aqi': 'AQI', 'city': 'Ville'},
            color='aqi',
            color_continuous_scale='RdYlBu_r'
        )
        fig_aqi.update_layout(height=400)
        st.plotly_chart(fig_aqi, use_container_width=True)

    with col2:
        # Distribution des catégories AQI
        aqi_counts = df_filtered['aqi_category'].value_counts()
        fig_aqi_cat = px.pie(
            values=aqi_counts.values,
            names=aqi_counts.index,
            title="Distribution des Catégories AQI"
        )
        fig_aqi_cat.update_layout(height=400)
        st.plotly_chart(fig_aqi_cat, use_container_width=True)

    # Analyse des polluants
    st.header("🔬 Analyse des Polluants")

    pollutants = ['pm2_5', 'pm10', 'co', 'no2', 'o3']

    col1, col2 = st.columns(2)

    with col1:
        # Heatmap des polluants
        pollutant_data = df_filtered.groupby('city')[pollutants].mean()
        fig_heatmap = px.imshow(
            pollutant_data.T,
            title="Concentrations Moyennes des Polluants",
            labels={'x': 'Ville', 'y': 'Polluant', 'color': 'Concentration'},
            aspect='auto'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col2:
        # Graphique radar pour une ville
        selected_city = st.selectbox("Choisir une ville pour l'analyse radar", cities)

        if selected_city:
            city_data = df_filtered[df_filtered['city'] == selected_city][pollutants].mean()

            fig_radar = go.Figure()

            fig_radar.add_trace(go.Scatterpolar(
                r=city_data.values,
                theta=pollutants,
                fill='toself',
                name=selected_city
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, city_data.max() * 1.1]
                    )
                ),
                showlegend=True,
                title=f"Profil des Polluants - {selected_city}",
                height=400
            )

            st.plotly_chart(fig_radar, use_container_width=True)

    # Conditions météo
    st.header("☁️ Conditions Météorologiques")

    col1, col2 = st.columns(2)

    with col1:
        # Distribution des conditions météo
        weather_counts = df_filtered['weather_main'].value_counts()
        fig_weather = px.bar(
            x=weather_counts.index,
            y=weather_counts.values,
            title="Distribution des Conditions Météo",
            labels={'x': 'Condition', 'y': 'Nombre d\'observations'}
        )
        fig_weather.update_layout(height=400)
        st.plotly_chart(fig_weather, use_container_width=True)

    with col2:
        # Corrélation température/humidité
        fig_scatter = px.scatter(
            df_filtered,
            x='temperature',
            y='humidity',
            color='city',
            size='wind_speed',
            title="Relation Température/Humidité",
            labels={'temperature': 'Température (°C)', 'humidity': 'Humidité (%)'}
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Tableau de données
    st.header("📋 Données Détaillées")

    # Options d'affichage
    show_raw_data = st.checkbox("Afficher les données brutes")

    if show_raw_data:
        st.subheader("Données filtrées")
        st.dataframe(df_filtered)

        # Bouton de téléchargement
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="Télécharger les données filtrées (CSV)",
            data=csv,
            file_name=f"weather_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # Statistiques avancées
    st.header("📊 Statistiques Avancées")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Statistiques Descriptives")
        numeric_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'aqi', 'comfort_score']
        st.dataframe(df_filtered[numeric_columns].describe())

    with col2:
        st.subheader("Corrélations")
        corr_matrix = df_filtered[numeric_columns].corr()
        fig_corr = px.imshow(
            corr_matrix,
            title="Matrice de Corrélation",
            labels={'color': 'Corrélation'},
            aspect='auto'
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

    # Ajoutez cette section après les "Statistiques Avancées" et avant les "Informations Pipeline ETL"

# Analyse de stabilité météorologique
st.header("🎯 Analyse de Stabilité Météorologique")

def calculate_stability_score(city_data):
    """Calcule un score de stabilité basé sur la variance des paramètres météo"""
    stability_factors = {
        'temperature': city_data['temperature'].std(),
        'humidity': city_data['humidity'].std(),
        'pressure': city_data['pressure'].std(),
        'wind_speed': city_data['wind_speed'].std(),
        'aqi': city_data['aqi'].std()
    }

    # Normalisation des scores (plus la variance est faible, plus c'est stable)
    normalized_scores = {}
    for factor, std_val in stability_factors.items():
        # Inverse de la variance normalisée (0-100, 100 = très stable)
        normalized_scores[factor] = max(0, 100 - (std_val * 10))

    # Score global (moyenne pondérée)
    weights = {
        'temperature': 0.3,
        'humidity': 0.2,
        'pressure': 0.2,
        'wind_speed': 0.15,
        'aqi': 0.15
    }

    global_score = sum(normalized_scores[factor] * weights[factor]
                      for factor in normalized_scores)

    return global_score, normalized_scores, stability_factors

# Calcul des scores de stabilité par ville
stability_results = {}
for city in df_filtered['city'].unique():
    city_data = df_filtered[df_filtered['city'] == city]
    if len(city_data) > 1:  # Besoin d'au moins 2 points pour calculer la variance
        score, details, raw_std = calculate_stability_score(city_data)
        stability_results[city] = {
            'score': score,
            'details': details,
            'raw_std': raw_std,
            'data_points': len(city_data)
        }

if stability_results:
    # Tri par score de stabilité
    sorted_cities = sorted(stability_results.items(), key=lambda x: x[1]['score'], reverse=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Classement des Villes les Plus Stables")

        # Affichage du podium
        for i, (city, data) in enumerate(sorted_cities[:3]):
            if i == 0:
                st.success(f"🥇 **{city}** - Score: {data['score']:.1f}/100")
                st.info(f"📊 **VILLE LA PLUS STABLE**: {city}")
            elif i == 1:
                st.info(f"🥈 **{city}** - Score: {data['score']:.1f}/100")
            else:
                st.warning(f"🥉 **{city}** - Score: {data['score']:.1f}/100")

        # Tableau complet des scores
        st.subheader("📊 Scores de Stabilité Détaillés")
        stability_df = pd.DataFrame({
            'Ville': [city for city, _ in sorted_cities],
            'Score Global': [data['score'] for _, data in sorted_cities],
            'Stabilité Température': [data['details']['temperature'] for _, data in sorted_cities],
            'Stabilité Humidité': [data['details']['humidity'] for _, data in sorted_cities],
            'Stabilité Pression': [data['details']['pressure'] for _, data in sorted_cities],
            'Stabilité Vent': [data['details']['wind_speed'] for _, data in sorted_cities],
            'Stabilité AQI': [data['details']['aqi'] for _, data in sorted_cities],
            'Nb Mesures': [data['data_points'] for _, data in sorted_cities]
        })

        st.dataframe(stability_df.round(1))

    with col2:
        st.subheader("📈 Visualisation des Scores")

        # Graphique en barres des scores globaux
        fig_stability = px.bar(
            x=[city for city, _ in sorted_cities],
            y=[data['score'] for _, data in sorted_cities],
            title="Scores de Stabilité Météorologique",
            labels={'x': 'Ville', 'y': 'Score de Stabilité'},
            color=[data['score'] for _, data in sorted_cities],
            color_continuous_scale='RdYlGn'
        )
        fig_stability.update_layout(height=400)
        st.plotly_chart(fig_stability, use_container_width=True)

        # Graphique radar pour la ville la plus stable
        if sorted_cities:
            most_stable_city = sorted_cities[0][0]
            most_stable_data = sorted_cities[0][1]

            st.subheader(f"🎯 Profil de Stabilité - {most_stable_city}")

            categories = ['Température', 'Humidité', 'Pression', 'Vent', 'AQI']
            values = [
                most_stable_data['details']['temperature'],
                most_stable_data['details']['humidity'],
                most_stable_data['details']['pressure'],
                most_stable_data['details']['wind_speed'],
                most_stable_data['details']['aqi']
            ]

            fig_radar_stability = go.Figure()

            fig_radar_stability.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=most_stable_city,
                line_color='green'
            ))

            fig_radar_stability.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title=f"Profil de Stabilité - {most_stable_city}",
                height=400
            )

            st.plotly_chart(fig_radar_stability, use_container_width=True)

    # Analyse détaillée de la ville la plus stable
    if sorted_cities:
        most_stable_city = sorted_cities[0][0]
        most_stable_score = sorted_cities[0][1]['score']

        st.subheader(f"🔍 Analyse Détaillée - {most_stable_city}")

        city_data = df_filtered[df_filtered['city'] == most_stable_city]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            temp_std = city_data['temperature'].std()
            st.metric(
                "Écart-type Température",
                f"{temp_std:.2f}°C",
                help="Plus faible = plus stable"
            )

        with col2:
            humidity_std = city_data['humidity'].std()
            st.metric(
                "Écart-type Humidité",
                f"{humidity_std:.2f}%",
                help="Plus faible = plus stable"
            )

        with col3:
            pressure_std = city_data['pressure'].std()
            st.metric(
                "Écart-type Pression",
                f"{pressure_std:.2f} hPa",
                help="Plus faible = plus stable"
            )

        with col4:
            aqi_std = city_data['aqi'].std()
            st.metric(
                "Écart-type AQI",
                f"{aqi_std:.2f}",
                help="Plus faible = plus stable"
            )

        # Recommandations
        st.subheader("💡 Recommandations")

        if most_stable_score >= 80:
            st.success(f"✅ **{most_stable_city}** présente une excellente stabilité météorologique (Score: {most_stable_score:.1f}/100)")
            st.info("🎯 Cette ville est idéale pour les activités nécessitant des conditions météo prévisibles")
        elif most_stable_score >= 60:
            st.info(f"✅ **{most_stable_city}** présente une bonne stabilité météorologique (Score: {most_stable_score:.1f}/100)")
            st.warning("⚠️ Surveillance recommandée pour certaines activités sensibles")
        else:
            st.warning(f"⚠️ **{most_stable_city}** présente une stabilité météorologique modérée (Score: {most_stable_score:.1f}/100)")
            st.error("🚨 Conditions météo variables, planification prudente recommandée")

else:
    st.warning("⚠️ Données insuffisantes pour calculer les scores de stabilité")
    # Informations sur le pipeline ETL
    st.header("⚙️ Informations Pipeline ETL")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"**Dernière mise à jour:** {df['timestamp'].max()}")

    with col2:
        st.info(f"**Nombre total d'enregistrements:** {len(df)}")

    with col3:
        st.info(f"**Villes surveillées:** {df['city'].nunique()}")

    # Status du pipeline (simulation)
    st.subheader("État du Pipeline Airflow")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.success("✅ Extraction OpenWeather")

    with col2:
        st.success("✅ Nettoyage des données")

    with col3:
        st.success("✅ Sauvegarde")

    with col4:
        st.success("✅ Merge des sources")

# Vérification des données météo principales
if 'weather_main' not in df.columns:
    st.warning("⚠️ Données météo principales manquantes")

    # Informations sur les données attendues
    st.header("📋 Format des Données Attendu")

    st.markdown("""
    Le dashboard attend un fichier CSV avec les colonnes suivantes :

    - **city**: Nom de la ville
    - **country**: Code pays
    - **lat, lon**: Coordonnées géographiques
    - **date**: Date de l'observation
    - **timestamp**: Horodatage précis
    - **temperature**: Température en °C
    - **feels_like**: Température ressentie
    - **humidity**: Humidité en %
    - **pressure**: Pression atmosphérique
    - **wind_speed**: Vitesse du vent
    - **wind_direction**: Direction du vent
    - **cloudiness**: Couverture nuageuse
    - **visibility**: Visibilité
    - **weather_main**: Condition météo principale
    - **aqi**: Indice de qualité de l'air
    - **pm2_5, pm10, co, no2, o3**: Polluants
    - **comfort_score**: Score de confort
    - **season**: Saison
    - **aqi_category**: Catégorie AQI
    """)

# Footer
st.markdown("---")
st.markdown("🌦️ Dashboard Météo - Pipeline ETL Automatisé avec Apache Airflow")
