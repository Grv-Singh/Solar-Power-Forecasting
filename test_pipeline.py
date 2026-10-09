import unittest
import numpy as np
import os
from dataset import load_pv_dataset, load_all_pv_datasets, prepare_features_target, create_sequences, get_train_test_data
from live_api import get_live_forecast_features, GERMAN_SOLAR_FARMS
from predict_live import predict_live
from xgboost_solar import train_and_evaluate_xgboost
from ensemble_solar import evaluate_ensemble
from app import app

class TestSolarPipeline(unittest.TestCase):

    def test_dataset_loading(self):
        df = load_pv_dataset('pv_01.csv')
        self.assertIn('power_normed', df.columns)
        self.assertNotIn('time_idx', df.columns)
        self.assertEqual(df.shape[1], 50)

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

    def test_predict_live_cli(self):
        res_ann = predict_live(plant_id=1, model_type='ann')
        self.assertEqual(res_ann['model_type'], 'ANN')
        self.assertIn('predicted_power_normed', res_ann)
        self.assertTrue(0.0 <= res_ann['predicted_power_normed'] <= 1.0)

    def test_xgboost_and_ensemble(self):
        model, metrics = train_and_evaluate_xgboost(plant_id=1, n_estimators=10)
        self.assertIn('rmse', metrics)

        ensemble_metrics = evaluate_ensemble(plant_id=1)
        self.assertIn('rmse', ensemble_metrics)

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
