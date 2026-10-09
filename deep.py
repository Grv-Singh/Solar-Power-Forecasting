import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataset import load_all_pv_datasets, get_train_test_data

def build_ann_model(input_dim, hidden_units=[64, 32, 16], activation='relu', output_activation='sigmoid'):
    """
    Builds a Sequential ANN model for regression.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    for units in hidden_units:
        model.add(tf.keras.layers.Dense(units=units, activation=activation))

    model.add(tf.keras.layers.Dense(units=1, activation=output_activation))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_and_evaluate_ann(plant_id='all', epochs=50, batch_size=32, pca_components=None):
    """
    Trains and evaluates ANN model for a specific plant or across all 21 plants.
    """
    print(f"\n==========================================")
    print(f"Training ANN Model (Plant ID: {plant_id}, PCA Components: {pca_components})")
    print(f"==========================================")

    data = get_train_test_data(
        plant_id=plant_id,
        test_size=0.2,
        pca_components=pca_components,
        scaler_type='standard'
    )

    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    scaler_y = data['scaler_y']

    model = build_ann_model(input_dim=X_train.shape[1], hidden_units=[64, 32, 16])

    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )

    # Predictions
    y_pred_scaled = model.predict(X_test)

    # Rescale back to original target space
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)

    print(f"\nANN Evaluation Results (Plant: {plant_id}):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R^2:  {r2:.4f}")

    return model, {'rmse': rmse, 'mae': mae, 'r2': r2}

def evaluate_all_plants_ann(epochs=20, batch_size=32, pca_components=10):
    """
    Evaluates ANN model individually across all 21 solar farms and reports summary metrics.
    """
    all_datasets = load_all_pv_datasets()
    plant_ids = sorted(all_datasets.keys())
    results = []

    print(f"\nEvaluating ANN across {len(plant_ids)} solar facilities...")
    for pid in plant_ids:
        _, metrics = train_and_evaluate_ann(
            plant_id=pid,
            epochs=epochs,
            batch_size=batch_size,
            pca_components=pca_components
        )
        results.append({'plant_id': pid, **metrics})

    df_res = pd.DataFrame(results)
    print("\nSummary of ANN Performance Across All 21 Solar Facilities:")
    print(df_res.to_string(index=False))
    print(f"\nMean RMSE: {df_res['rmse'].mean():.4f}")
    print(f"Mean MAE:  {df_res['mae'].mean():.4f}")
    print(f"Mean R^2:  {df_res['r2'].mean():.4f}")
    return df_res

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Solar Power Forecasting ANN model.")
    parser.add_argument('--plant_id', type=str, default='all', help="PV Plant ID (1-21) or 'all' or 'eval_all'")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--pca', type=int, default=None, help="Number of PCA components (optional)")
    parser.add_argument('--save_model', type=str, default='solar_ann_model.keras', help="Output model path")

    args = parser.parse_args()

    if args.plant_id == 'eval_all':
        evaluate_all_plants_ann(epochs=args.epochs, batch_size=args.batch_size, pca_components=args.pca)
    else:
        plant_id = int(args.plant_id) if args.plant_id.isdigit() else args.plant_id
        model, metrics = train_and_evaluate_ann(
            plant_id=plant_id,
            epochs=args.epochs,
            batch_size=args.batch_size,
            pca_components=args.pca
        )
        if args.save_model:
            model.save(args.save_model)
            print(f"ANN Model saved successfully to {args.save_model}")

if __name__ == "__main__":
    main()
