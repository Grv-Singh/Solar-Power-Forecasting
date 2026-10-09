import os
import argparse
import tensorflow as tf
import numpy as np
from live_api import get_live_forecast_features, GERMAN_SOLAR_FARMS
from dataset import get_train_test_data

def load_or_train_ann():
    model_path = 'solar_ann_model.keras'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        print("Training default ANN model...")
        from deep import train_and_evaluate_ann
        model, _ = train_and_evaluate_ann(plant_id='all', epochs=10)
        model.save(model_path)
    return model

def load_or_train_lstm():
    model_path = 'solar_lstm_model.keras'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        print("Training default LSTM model...")
        from lstm_solar import train_and_evaluate_lstm
        model, _ = train_and_evaluate_lstm(plant_id='all', epochs=10)
        model.save(model_path)
    return model

def predict_live(plant_id=1, model_type='ann'):
    """
    Fetches live API weather/irradiance data and predicts solar power normalized output.
    """
    data = get_train_test_data(plant_id=plant_id, scaler_type='minmax' if model_type == 'lstm' else 'standard')
    scaler_X = data['scaler_X']
    scaler_y = data['scaler_y']

    if model_type == 'lstm':
        features, timestamps, farm_info = get_live_forecast_features(plant_id=plant_id, time_steps=8)
        model = load_or_train_lstm()
        # Scale features
        scaled_features = scaler_X.transform(features)
        # Sequence input shape (1, 8, 49)
        input_seq = np.expand_dims(scaled_features, axis=0)
        pred_scaled = model.predict(input_seq, verbose=0)
        pred_val = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
        timestamp = timestamps[-1]
    else:
        features, timestamps, farm_info = get_live_forecast_features(plant_id=plant_id, time_steps=1)
        model = load_or_train_ann()
        scaled_features = scaler_X.transform(features)
        pred_scaled = model.predict(scaled_features, verbose=0)
        pred_val = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])
        timestamp = timestamps[0]

    # Normalized power is bounded [0, 1]
    pred_val = float(np.clip(pred_val, 0.0, 1.0))

    return {
        'plant_id': plant_id,
        'plant_name': farm_info['name'],
        'latitude': farm_info['lat'],
        'longitude': farm_info['lon'],
        'timestamp': timestamp,
        'model_type': model_type.upper(),
        'predicted_power_normed': round(pred_val, 4),
        'predicted_power_percentage': f"{pred_val * 100:.2f}%"
    }

def main():
    parser = argparse.ArgumentParser(description="Live Solar Power Forecasting CLI Tool")
    parser.add_argument('--plant_id', type=int, default=1, help="PV Plant ID (1-21)")
    parser.add_argument('--model', type=str, choices=['ann', 'lstm'], default='ann', help="Model architecture")

    args = parser.parse_args()
    res = predict_live(plant_id=args.plant_id, model_type=args.model)

    print("\n==========================================")
    print("LIVE SOLAR POWER FORECAST RESULT")
    print("==========================================")
    for k, v in res.items():
        print(f"  {k:30s}: {v}")
    print("==========================================\n")

if __name__ == "__main__":
    main()
