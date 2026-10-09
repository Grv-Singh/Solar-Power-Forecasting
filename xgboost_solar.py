import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dataset import get_train_test_data

def train_and_evaluate_xgboost(plant_id='all', n_estimators=100, max_depth=6, learning_rate=0.1):
    """
    Trains and evaluates XGBoost regressor for solar power forecasting.
    """
    print(f"\n==========================================")
    print(f"Training XGBoost Model (Plant ID: {plant_id})")
    print(f"==========================================")

    data = get_train_test_data(
        plant_id=plant_id,
        test_size=0.2,
        scaler_type='standard'
    )

    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    scaler_y = data['scaler_y']

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred_scaled = model.predict(X_test)

    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)

    print(f"\nXGBoost Evaluation Results (Plant: {plant_id}):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R^2:  {r2:.4f}")

    return model, {'rmse': rmse, 'mae': mae, 'r2': r2}

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost Solar Forecasting Baseline.")
    parser.add_argument('--plant_id', type=str, default='all', help="PV Plant ID (1-21) or 'all'")
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=6)

    args = parser.parse_args()
    plant_id = int(args.plant_id) if args.plant_id.isdigit() else args.plant_id
    train_and_evaluate_xgboost(plant_id=plant_id, n_estimators=args.n_estimators, max_depth=args.max_depth)

if __name__ == "__main__":
    main()
