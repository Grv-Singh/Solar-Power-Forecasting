import numpy as np
from deep import train_and_evaluate_ann
from xgboost_solar import train_and_evaluate_xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dataset import get_train_test_data

def evaluate_ensemble(plant_id='all'):
    """
    Evaluates a weighted Ensemble predictor combining ANN and XGBoost forecasts.
    """
    print(f"\n==========================================")
    print(f"Evaluating Ensemble Forecasting (Plant ID: {plant_id})")
    print(f"==========================================")

    data = get_train_test_data(plant_id=plant_id, test_size=0.2, scaler_type='standard')
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    scaler_y = data['scaler_y']

    ann_model, _ = train_and_evaluate_ann(plant_id=plant_id, epochs=10)
    xgb_model, _ = train_and_evaluate_xgboost(plant_id=plant_id, n_estimators=50)

    pred_ann_scaled = ann_model.predict(X_test, verbose=0).flatten()
    pred_xgb_scaled = xgb_model.predict(X_test).flatten()

    # 50/50 Ensemble average
    pred_ensemble_scaled = 0.5 * pred_ann_scaled + 0.5 * pred_xgb_scaled

    y_pred = scaler_y.inverse_transform(pred_ensemble_scaled.reshape(-1, 1)).flatten()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)

    print(f"\nEnsemble Evaluation Results (ANN + XGBoost):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R^2:  {r2:.4f}")

    return {'rmse': rmse, 'mae': mae, 'r2': r2}

if __name__ == "__main__":
    evaluate_ensemble(plant_id=1)
