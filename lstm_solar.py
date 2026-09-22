import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataset import load_all_pv_datasets, get_train_test_data

def build_lstm_model(input_shape, lstm_units=[64, 32], dropout_rate=0.2):
    """
    Builds a Sequential LSTM model for time series regression.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)
        model.add(tf.keras.layers.LSTM(units=units, return_sequences=return_sequences))
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_and_evaluate_lstm(plant_id='all', sequence_length=8, epochs=30, batch_size=32, pca_components=None):
    """
    Trains and evaluates LSTM model for a specific plant or across all plants.
    """
    print(f"\n==========================================")
    print(f"Training LSTM Model (Plant ID: {plant_id}, Seq Length: {sequence_length}, PCA: {pca_components})")
    print(f"==========================================")

    data = get_train_test_data(
        plant_id=plant_id,
        test_size=0.2,
        sequence_length=sequence_length,
        pca_components=pca_components,
        scaler_type='minmax'
    )

    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    scaler_y = data['scaler_y']

    # Input shape for LSTM: (time_steps, features)
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape=input_shape, lstm_units=[64, 32], dropout_rate=0.2)

    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1,
        shuffle=True
    )

    # Predict
    y_pred_scaled = model.predict(X_test)

    # Inverse scale to original domain
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)

    print(f"\nLSTM Evaluation Results (Plant: {plant_id}):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R^2:  {r2:.4f}")

    return model, {'rmse': rmse, 'mae': mae, 'r2': r2}

def evaluate_all_plants_lstm(sequence_length=8, epochs=15, batch_size=32):
    """
    Evaluates LSTM model across all 21 PV facilities.
    """
    all_datasets = load_all_pv_datasets()
    plant_ids = sorted(all_datasets.keys())
    results = []

    print(f"\nEvaluating LSTM across {len(plant_ids)} solar facilities...")
    for pid in plant_ids:
        _, metrics = train_and_evaluate_lstm(
            plant_id=pid,
            sequence_length=sequence_length,
            epochs=epochs,
            batch_size=batch_size
        )
        results.append({'plant_id': pid, **metrics})

    df_res = pd.DataFrame(results)
    print("\nSummary of LSTM Performance Across All 21 Solar Facilities:")
    print(df_res.to_string(index=False))
    print(f"\nMean RMSE: {df_res['rmse'].mean():.4f}")
    print(f"Mean MAE:  {df_res['mae'].mean():.4f}")
    print(f"Mean R^2:  {df_res['r2'].mean():.4f}")
    return df_res

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Solar Power Forecasting LSTM model.")
    parser.add_argument('--plant_id', type=str, default='all', help="PV Plant ID (1-21) or 'all' or 'eval_all'")
    parser.add_argument('--seq_length', type=int, default=8, help="Sequence length (time steps)")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--pca', type=int, default=None, help="Number of PCA components (optional)")
    parser.add_argument('--save_model', type=str, default='solar_lstm_model.keras', help="Output model path")

    args = parser.parse_args()

    if args.plant_id == 'eval_all':
        evaluate_all_plants_lstm(sequence_length=args.seq_length, epochs=args.epochs, batch_size=args.batch_size)
    else:
        plant_id = int(args.plant_id) if args.plant_id.isdigit() else args.plant_id
        model, metrics = train_and_evaluate_lstm(
            plant_id=plant_id,
            sequence_length=args.seq_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            pca_components=args.pca
        )
        if args.save_model:
            model.save(args.save_model)
            print(f"LSTM Model saved successfully to {args.save_model}")

if __name__ == "__main__":
    main()
