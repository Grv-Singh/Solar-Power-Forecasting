import unittest
import numpy as np
import os
from dataset import load_pv_dataset, load_all_pv_datasets, prepare_features_target, create_sequences, get_train_test_data
from live_api import get_live_forecast_features, GERMAN_SOLAR_FARMS
from predict_live import predict_live
from app import app

class TestSolarPipeline(unittest.TestCase):

    def test_dataset_loading(self):
        df = load_pv_dataset('pv_01.csv')
        self.assertIn('power_normed', df.columns)
        self.assertNotIn('time_idx', df.columns)
        self.assertEqual(df.shape[1], 50)  # 49 features + 1 target

    def test_all_datasets_loading(self):
        datasets = load_all_pv_datasets()
        self.assertEqual(len(datasets), 21)
        self.assertIn(1, datasets)
        self.assertIn(21, datasets)

    def test_pca_preparation(self):
        df = load_pv_dataset('pv_01.csv')
        X, y, pca = prepare_features_target(df, pca_components=10)
        self.assertEqual(X.shape[1], 10)
        self.assertIsNotNone(pca)

    def test_sequence_creation(self):
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        Xs, ys = create_sequences(X, y, time_steps=8)
        self.assertEqual(Xs.shape, (92, 8, 10))
        self.assertEqual(ys.shape, (92,))

    def test_live_api_fetching(self):
        features, timestamps, farm_info = get_live_forecast_features(plant_id=1, time_steps=8)
        self.assertEqual(features.shape, (8, 49))
        self.assertEqual(len(timestamps), 8)
        self.assertEqual(farm_info['lat'], GERMAN_SOLAR_FARMS[1]['lat'])

    def test_open_meteo_helper_functions(self):
        from live_api import (
            _parse_time_features,
            _extract_raw_weather,
            _normalize_weather_features,
            _calculate_solar_position,
            build_feature_vector_from_open_meteo,
        )

        # Test time parsing helper
        time_feats = _parse_time_features("2026-06-15T12:00")
        self.assertEqual(time_feats['hour'], 12)
        self.assertAlmostEqual(time_feats['hour_norm'], 12 / 23.0)

        # Test raw weather extraction helper with missing keys (defaults fallback)
        dummy_hourly = {"time": ["2026-06-15T12:00"]}
        raw = _extract_raw_weather(dummy_hourly, 0)
        self.assertEqual(raw['temp'], 15.0)
        self.assertEqual(raw['rh'], 50.0)

        # Test normalization helper
        raw_custom = {
            'temp': 20.0,
            'rh': 60.0,
            'dew': 10.0,
            'press': 1000.0,
            'cloud': 50.0,
            'snow': 0.0,
            'dni': 500.0,
            'diffuse': 250.0,
            'sw': 500.0,
            'ws10': 10.0,
            'wd10': 180.0,
            'ws100': 15.0,
            'wd100': 180.0,
        }
        norm = _normalize_weather_features(raw_custom)
        self.assertAlmostEqual(norm['temp_norm'], (20.0 + 20) / 60.0)
        self.assertAlmostEqual(norm['global_norm'], 0.5)

        # Test solar position helper
        solar = _calculate_solar_position(12, 0.5)
        self.assertGreater(solar['solar_height'], 0.0)
        self.assertAlmostEqual(solar['extra_terr'], 0.45)

        # Test full feature vector construction from dictionary
        vec, ts = build_feature_vector_from_open_meteo(dummy_hourly, time_idx=0)
        self.assertEqual(vec.shape, (49,))
        self.assertEqual(ts, "2026-06-15T12:00")
        self.assertTrue(np.all(np.isfinite(vec)))

    def test_predict_live_cli(self):
        res_ann = predict_live(plant_id=1, model_type='ann')
        self.assertEqual(res_ann['model_type'], 'ANN')
        self.assertIn('predicted_power_normed', res_ann)
        self.assertTrue(0.0 <= res_ann['predicted_power_normed'] <= 1.0)

        res_lstm = predict_live(plant_id=1, model_type='lstm')
        self.assertEqual(res_lstm['model_type'], 'LSTM')
        self.assertIn('predicted_power_normed', res_lstm)

    def test_flask_app_endpoints(self):
        client = app.test_client()
        r1 = client.get('/')
        self.assertEqual(r1.status_code, 200)

        r2 = client.get('/farms')
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.get_json()['count'], 21)

        r3 = client.get('/predict?plant_id=2&model=ann')
        self.assertEqual(r3.status_code, 200)
        self.assertTrue(r3.get_json()['success'])

if __name__ == '__main__':
    unittest.main()
