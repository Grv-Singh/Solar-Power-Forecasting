import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

def load_data(file_path):
    """
    Loads the solar power dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Loading dataset from {file_path}...")
    try:
        # Try semicolon separator first (as seen in pv_01.csv)
        dataset = pd.read_csv(file_path, sep=';')
        if len(dataset.columns) <= 1:
             # Fallback to comma if semicolon didn't parse correctly
             dataset = pd.read_csv(file_path, sep=',')
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

    # Drop index column and empty trailing column if they exist
    if 'time_idx' in dataset.columns:
        dataset = dataset.drop(columns=['time_idx'])
    if 'Unnamed: 51' in dataset.columns:
        dataset = dataset.drop(columns=['Unnamed: 51'])

    # Check for target column
    if 'power_normed' not in dataset.columns:
        raise ValueError("Column 'power_normed' not found in dataset.")

    return dataset

def preprocess_data_ann(dataset, test_size=0.2, random_state=0):
    """
    Preprocesses data for ANN (MLP) models.

    Args:
        dataset (pd.DataFrame): The dataset.
        test_size (float): Fraction of data to use for testing.
        random_state (int): Random seed.

    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    X = dataset.drop(columns=['power_normed']).values
    y = dataset['power_normed'].values

    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def create_sequences(X, y, time_steps=1):
    """
    Creates sequences for LSTM.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def preprocess_data_lstm(dataset, split_ratio=0.8, time_steps=8):
    """
    Preprocesses data for LSTM models.

    Args:
        dataset (pd.DataFrame): The dataset.
        split_ratio (float): Fraction of data to use for training (sequential split).
        time_steps (int): Number of time steps for sequence.

    Returns:
        tuple: X_train, y_train, X_test, y_test, scaler_X, scaler_y
    """
    features = dataset.drop(columns=['power_normed']).values
    target = dataset['power_normed'].values.reshape(-1, 1)

    # Split into train and test - Time Series Split (no random shuffle)
    train_size = int(len(dataset) * split_ratio)

    # LSTM usually works better with MinMaxScaling
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

    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, time_steps)
    X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, time_steps)

    return X_train, y_train, X_test, y_test, scaler_X, scaler_y
