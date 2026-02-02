import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def main():
    print("Loading dataset...")
    # Load dataset with correct delimiter
    dataset = pd.read_csv('pv_01.csv', sep=';')

    # Drop index column and empty trailing column
    if 'time_idx' in dataset.columns:
        dataset = dataset.drop(columns=['time_idx'])
    if 'Unnamed: 51' in dataset.columns:
        dataset = dataset.drop(columns=['Unnamed: 51'])

    # Check for target column
    if 'power_normed' not in dataset.columns:
        raise ValueError("Column 'power_normed' not found.")

    features = dataset.drop(columns=['power_normed']).values
    target = dataset['power_normed'].values.reshape(-1, 1)

    # Split into train and test - Time Series Split (no random shuffle)
    train_size = int(len(dataset) * 0.8)

    # LSTM usually works better with scaling
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit on training data only to avoid data leakage
    X_train_raw = features[:train_size]
    X_test_raw = features[train_size:]
    y_train_raw = target[:train_size]
    y_test_raw = target[train_size:]

    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)

    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    # Create sequences
    # Using 8 steps (approx 1 day given 3h resolution)
    TIME_STEPS = 8

    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, TIME_STEPS)
    X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, TIME_STEPS)

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

    print("Starting training...")
    # Shuffle=False is often used for stateful LSTMs, but here samples are created as windows.
    # Shuffling windows is fine and helps training convergence.
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
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

    print(f"\nResults:")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R^2 Score: {r2}")

    # Save the model
    model.save('solar_lstm_model.keras')
    print("Model saved to solar_lstm_model.keras")

if __name__ == "__main__":
    main()
