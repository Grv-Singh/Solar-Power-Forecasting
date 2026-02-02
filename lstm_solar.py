import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import data_loader
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description='Train LSTM model for Solar Power Forecasting')
    parser.add_argument('--file', type=str, default='pv_01.csv', help='Path to the CSV dataset file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--time_steps', type=int, default=8, help='Time steps for LSTM sequence')
    args = parser.parse_args()

    # Load and preprocess
    try:
        dataset = data_loader.load_data(args.file)
        X_train, y_train, X_test, y_test, scaler_X, scaler_y = data_loader.preprocess_data_lstm(
            dataset, time_steps=args.time_steps
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Build LSTM model
    model = tf.keras.models.Sequential()
    # Input shape: (time_steps, features)
    model.add(tf.keras.layers.LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=32, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    print(f"Starting training on {args.file}...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=1,
        shuffle=True
    )

    # Predict
    print("Evaluating...")
    y_pred_scaled = model.predict(X_test)

    # Inverse transform to get actual values
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_inv = scaler_y.inverse_transform(y_test)

    # Evaluate
    mse = mean_squared_error(y_test_inv, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_pred)

    print(f"\nResults for {args.file}:")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R^2 Score: {r2}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inv[:100], label='Actual')
    plt.plot(y_pred[:100], label='Predicted')
    plt.title(f'LSTM: Actual vs Predicted Power (First 100 samples) - {args.file}')
    plt.xlabel('Sample')
    plt.ylabel('Normalized Power')
    plt.legend()
    plot_filename = f'lstm_results_{os.path.basename(args.file).split(".")[0]}.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

if __name__ == "__main__":
    main()
