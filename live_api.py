import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime

# Approximate coordinates (lat, lon) for 21 German Solar Farm locations
GERMAN_SOLAR_FARMS = {
    1: {"name": "Solar Farm 1 (Northern Germany)", "lat": 53.55, "lon": 9.99},
    2: {"name": "Solar Farm 2 (Brandenburg)", "lat": 52.52, "lon": 13.40},
    3: {"name": "Solar Farm 3 (Saxony)", "lat": 51.05, "lon": 13.73},
    4: {"name": "Solar Farm 4 (Bavaria)", "lat": 48.13, "lon": 11.58},
    5: {"name": "Solar Farm 5 (Baden-Württemberg)", "lat": 48.77, "lon": 9.18},
    6: {"name": "Solar Farm 6 (Hesse)", "lat": 50.11, "lon": 8.68},
    7: {"name": "Solar Farm 7 (North Rhine-Westphalia)", "lat": 51.22, "lon": 6.77},
    8: {"name": "Solar Farm 8 (Lower Saxony)", "lat": 52.37, "lon": 9.73},
    9: {"name": "Solar Farm 9 (Schleswig-Holstein)", "lat": 54.32, "lon": 10.13},
    10: {"name": "Solar Farm 10 (Mecklenburg-Vorpommern)", "lat": 53.63, "lon": 11.41},
    11: {"name": "Solar Farm 11 (Rhineland-Palatinate)", "lat": 49.99, "lon": 8.24},
    12: {"name": "Solar Farm 12 (Thuringia)", "lat": 50.98, "lon": 11.02},
    13: {"name": "Solar Farm 13 (Saxony-Anhalt)", "lat": 52.12, "lon": 11.62},
    14: {"name": "Solar Farm 14 (Saarland)", "lat": 49.23, "lon": 6.99},
    15: {"name": "Solar Farm 15 (Bavaria South)", "lat": 47.80, "lon": 11.00},
    16: {"name": "Solar Farm 16 (Bavaria East)", "lat": 48.90, "lon": 12.10},
    17: {"name": "Solar Farm 17 (Baden-Württemberg West)", "lat": 48.00, "lon": 7.85},
    18: {"name": "Solar Farm 18 (Brandenburg South)", "lat": 51.75, "lon": 14.33},
    19: {"name": "Solar Farm 19 (Lower Saxony West)", "lat": 53.14, "lon": 8.21},
    20: {"name": "Solar Farm 20 (North Rhine-Westphalia North)", "lat": 51.96, "lon": 7.62},
    21: {"name": "Solar Farm 21 (Hesse North)", "lat": 51.31, "lon": 9.49},
}

def fetch_open_meteo_live_data(lat, lon):
    """
    Fetches real-time / hourly forecast data from Open-Meteo API for given lat/lon.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,cloud_cover,"
        f"snow_depth,direct_normal_irradiance,diffuse_radiation,shortwave_radiation,"
        f"wind_speed_10m,wind_direction_10m,wind_speed_100m,wind_direction_100m"
        f"&timezone=UTC"
    )

    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(f"Open-Meteo API request failed with status code {response.status_code}: {response.text}")

    return response.json()

def _parse_time_features(time_str):
    """
    Parses ISO time string and returns normalized/cyclical hour, month, and season features.
    """
    dt = datetime.fromisoformat(time_str)
    hour = dt.hour
    month = dt.month

    hour_norm = hour / 23.0
    hour_cos = math.cos(2 * math.pi * hour / 24.0)
    hour_sin = math.sin(2 * math.pi * hour / 24.0)

    month_norm = (month - 1) / 11.0
    month_cos = math.cos(2 * math.pi * month / 12.0)
    month_sin = math.sin(2 * math.pi * month / 12.0)

    season = (month % 12 + 3) // 3
    season_norm = season / 4.0
    season_cos = math.cos(2 * math.pi * season / 4.0)
    season_sin = math.sin(2 * math.pi * season / 4.0)

    return {
        'hour': hour,
        'hour_norm': hour_norm,
        'hour_cos': hour_cos,
        'hour_sin': hour_sin,
        'month_norm': month_norm,
        'month_cos': month_cos,
        'month_sin': month_sin,
        'season_norm': season_norm,
        'season_cos': season_cos,
        'season_sin': season_sin,
    }

def _extract_raw_weather(hourly_data, time_idx):
    """
    Extracts weather variables from hourly Open-Meteo dictionary with default fallbacks.
    """
    return {
        'temp': hourly_data['temperature_2m'][time_idx] if 'temperature_2m' in hourly_data else 15.0,
        'rh': hourly_data['relative_humidity_2m'][time_idx] if 'relative_humidity_2m' in hourly_data else 50.0,
        'dew': hourly_data['dew_point_2m'][time_idx] if 'dew_point_2m' in hourly_data else 10.0,
        'press': hourly_data['surface_pressure'][time_idx] if 'surface_pressure' in hourly_data else 1013.25,
        'cloud': hourly_data['cloud_cover'][time_idx] if 'cloud_cover' in hourly_data else 20.0,
        'snow': hourly_data['snow_depth'][time_idx] if 'snow_depth' in hourly_data else 0.0,
        'dni': hourly_data['direct_normal_irradiance'][time_idx] if 'direct_normal_irradiance' in hourly_data else 0.0,
        'diffuse': hourly_data['diffuse_radiation'][time_idx] if 'diffuse_radiation' in hourly_data else 0.0,
        'sw': hourly_data['shortwave_radiation'][time_idx] if 'shortwave_radiation' in hourly_data else 0.0,
        'ws10': hourly_data['wind_speed_10m'][time_idx] if 'wind_speed_10m' in hourly_data else 3.0,
        'wd10': hourly_data['wind_direction_10m'][time_idx] if 'wind_direction_10m' in hourly_data else 180.0,
        'ws100': hourly_data['wind_speed_100m'][time_idx] if 'wind_speed_100m' in hourly_data else 5.0,
        'wd100': hourly_data['wind_direction_100m'][time_idx] if 'wind_direction_100m' in hourly_data else 180.0,
    }

def _normalize_weather_features(raw):
    """
    Normalizes raw weather inputs to match dataset bounds and computes wind components.
    """
    wd10_rad = math.radians(raw['wd10'])
    wd100_rad = math.radians(raw['wd100'])

    ws10_norm = np.clip(raw['ws10'] / 25.0, 0, 1)
    ws100_norm = np.clip(raw['ws100'] / 35.0, 0, 1)
    wd10_cos = math.cos(wd10_rad)
    wd10_sin = math.sin(wd10_rad)
    wd100_cos = math.cos(wd100_rad)
    wd100_sin = math.sin(wd100_rad)

    return {
        'temp_norm': np.clip((raw['temp'] + 20) / 60.0, 0, 1),
        'rh_norm': np.clip(raw['rh'] / 100.0, 0, 1),
        'dew_norm': np.clip((raw['dew'] + 20) / 40.0, 0, 1),
        'press_norm': np.clip((raw['press'] - 950) / 100.0, 0, 1),
        'cloud_norm': np.clip(raw['cloud'] / 100.0, 0, 1),
        'snow_norm': np.clip(raw['snow'] / 1.0, 0, 1),
        'dni_norm': np.clip(raw['dni'] / 1000.0, 0, 1),
        'diffuse_norm': np.clip(raw['diffuse'] / 500.0, 0, 1),
        'global_norm': np.clip(raw['sw'] / 1000.0, 0, 1),
        'ws10_norm': ws10_norm,
        'wd10_deg': raw['wd10'],
        'wd10_cos': wd10_cos,
        'wd10_sin': wd10_sin,
        'ws100_norm': ws100_norm,
        'wd100_deg': raw['wd100'],
        'wd100_cos': wd100_cos,
        'wd100_sin': wd100_sin,
    }

def _calculate_solar_position(hour, global_norm):
    """
    Approximates solar height, zenith, azimuth, and extraterrestrial radiation.
    """
    solar_height = max(0.0, math.sin(math.radians(max(0, (12 - abs(hour - 12)) * 15))))
    theta_z = 1.0 - solar_height
    solar_azimuth = hour / 24.0
    extra_terr = global_norm * 0.9

    return {
        'solar_height': solar_height,
        'theta_z': theta_z,
        'solar_azimuth': solar_azimuth,
        'extra_terr': extra_terr,
    }

def build_feature_vector_from_open_meteo(hourly_data, time_idx=0):
    """
    Constructs a normalized 49-feature vector matching dataset.py column order
    from Open-Meteo hourly API response.
    """
    time_str = hourly_data['time'][time_idx]
    time_feats = _parse_time_features(time_str)
    raw_weather = _extract_raw_weather(hourly_data, time_idx)
    norm_weather = _normalize_weather_features(raw_weather)
    solar_feats = _calculate_solar_position(time_feats['hour'], norm_weather['global_norm'])

    feature_dict = {
        'hour_of_day': time_feats['hour_norm'],
        'hour_of_day_cos': time_feats['hour_cos'],
        'hour_of_day_sin': time_feats['hour_sin'],
        'month_of_year': time_feats['month_norm'],
        'month_of_year_cos': time_feats['month_cos'],
        'month_of_year_sin': time_feats['month_sin'],
        'season_of_year': time_feats['season_norm'],
        'season_of_year_cos': time_feats['season_cos'],
        'season_of_year_sin': time_feats['season_sin'],
        'sunposition_thetaZ': solar_feats['theta_z'],
        'sunposition_solarAzimuth': solar_feats['solar_azimuth'],
        'sunposition_extraTerr': solar_feats['extra_terr'],
        'sunposition_solarHeight': solar_feats['solar_height'],
        'clearsky_diffuse': norm_weather['diffuse_norm'],
        'clearsky_direct': norm_weather['dni_norm'],
        'clearsky_global': norm_weather['global_norm'],
        'clearsky_diffuse_agg': norm_weather['diffuse_norm'],
        'clearsky_direct_agg': norm_weather['dni_norm'],
        'clearsky_global_agg': norm_weather['global_norm'],
        'Albedo': 0.15,
        'WindComponentUat0': norm_weather['ws10_norm'] * norm_weather['wd10_sin'],
        'WindComponentVat0': norm_weather['ws10_norm'] * norm_weather['wd10_cos'],
        'WindComponentUat100': norm_weather['ws100_norm'] * norm_weather['wd100_sin'],
        'WindComponentVat100': norm_weather['ws100_norm'] * norm_weather['wd100_cos'],
        'DewpointTemperatureAt0': norm_weather['dew_norm'],
        'TemperatureAt0': norm_weather['temp_norm'],
        'PotentialVorticityAt1000': 0.5,
        'PotentialVorticityAt950': 0.5,
        'RelativeHumidityAt1000': norm_weather['rh_norm'],
        'RelativeHumidityAt950': norm_weather['rh_norm'],
        'RelativeHumidityAt0': norm_weather['rh_norm'],
        'SnowDensityAt0': norm_weather['snow_norm'] * 0.3,
        'SnowDepthAt0': norm_weather['snow_norm'],
        'SnowfallPlusStratiformSurfaceAt0': 0.0,
        'SurfacePressureAt0': norm_weather['press_norm'],
        'SolarRadiationGlobalAt0': norm_weather['global_norm'],
        'SolarRadiationDirectAt0': norm_weather['dni_norm'],
        'SolarRadiationDiffuseAt0': norm_weather['diffuse_norm'],
        'TotalCloudCoverAt0': norm_weather['cloud_norm'],
        'LowerWindSpeed': norm_weather['ws10_norm'],
        'LowerWindDirection': norm_weather['wd10_deg'] / 360.0,
        'LowerWindDirectionMath': norm_weather['wd10_deg'] / 360.0,
        'LowerWindDirectionCos': norm_weather['wd10_cos'],
        'LowerWindDirectionSin': norm_weather['wd10_sin'],
        'UpperWindSpeed': norm_weather['ws100_norm'],
        'UpperWindDirection': norm_weather['wd100_deg'] / 360.0,
        'UpperWindDirectionMath': norm_weather['wd100_deg'] / 360.0,
        'UpperWindDirectionCos': norm_weather['wd100_cos'],
        'UpperWindDirectionSin': norm_weather['wd100_sin']
    }

    return np.array(list(feature_dict.values()), dtype=np.float32), time_str

def get_live_forecast_features(plant_id=1, time_steps=1):
    """
    Fetches live forecast features for a specific German PV plant.
    Returns feature array for model inference.
    """
    if plant_id not in GERMAN_SOLAR_FARMS:
        plant_id = 1

    farm_info = GERMAN_SOLAR_FARMS[plant_id]
    api_res = fetch_open_meteo_live_data(farm_info['lat'], farm_info['lon'])
    hourly = api_res['hourly']

    features_list = []
    timestamps = []

    for t_idx in range(min(time_steps, len(hourly['time']))):
        vec, ts = build_feature_vector_from_open_meteo(hourly, time_idx=t_idx)
        features_list.append(vec)
        timestamps.append(ts)

    return np.array(features_list), timestamps, farm_info

if __name__ == "__main__":
    print("Testing live API data fetching for Solar Plant 1...")
    features, timestamps, farm_info = get_live_forecast_features(plant_id=1, time_steps=8)
    print(f"Farm Info: {farm_info}")
    print(f"Fetched {len(timestamps)} time steps. Feature shape: {features.shape}")
    print(f"Latest Timestamp: {timestamps[0]}")
