import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def load_pv_dataset(filepath):
    """
    Loads a single PV dataset CSV file and cleans unwanted columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist.")

    # Try semicolon delimiter first (pv_01.csv to pv_21.csv format)
    df = pd.read_csv(filepath, sep=';')
    if 'power_normed' not in df.columns:
        # Fallback to comma delimiter if needed (e.g. pv_1.csv)
        df = pd.read_csv(filepath, sep=',')

    # Drop index and empty columns if present
    columns_to_drop = [c for c in df.columns if c in ['time_idx', 'Unnamed: 51'] or c.startswith('Unnamed')]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    return df

def load_all_pv_datasets(data_dir='.'):
    """
    Loads all valid pv_*.csv datasets in the given directory.
    Excludes pv_1.csv if it does not contain target 'power_normed'.
    Returns a dictionary mapping plant_id (int) to DataFrame.
    """
    pv_files = sorted(glob.glob(os.path.join(data_dir, 'pv_*.csv')))
    datasets = {}

    for filepath in pv_files:
        filename = os.path.basename(filepath)
        # Handle pv_01.csv to pv_21.csv vs pv_1.csv
        try:
            df = load_pv_dataset(filepath)
            if 'power_normed' in df.columns:
                # Extract Plant ID number
                plant_id_str = filename.replace('pv_', '').replace('.csv', '')
                plant_id = int(plant_id_str)
                datasets[plant_id] = df
        except Exception as e:
            print(f"Warning: Failed to load {filename}: {e}")

    return datasets

def prepare_features_target(df, pca_components=None):
    """
    Extracts feature matrix X and target array y from a plant DataFrame.
    Optionally applies PCA to reduce feature dimensions.
    Returns X, y, pca_object (or None).
    """
    if 'power_normed' not in df.columns:
        raise ValueError("DataFrame does not contain target column 'power_normed'.")

    X = df.drop(columns=['power_normed']).values
    y = df['power_normed'].values

    pca = None
    if pca_components is not None and pca_components > 0:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)

    return X, y, pca

def create_sequences(X, y, time_steps=8):
    """
    Creates time series input sequences for LSTM models.
    Returns 3D feature array Xs (samples, time_steps, features) and 1D target array ys.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def get_train_test_data(plant_id=1, data_dir='.', test_size=0.2, sequence_length=None, pca_components=None, scaler_type='standard'):
    """
    High-level function to load, split, and scale data for a specific plant or aggregated plants.
    """
    if plant_id == 'all':
        datasets = load_all_pv_datasets(data_dir=data_dir)
        dfs = [datasets[k] for k in sorted(datasets.keys())]
        combined_df = pd.concat(dfs, ignore_index=True)
        X, y, pca = prepare_features_target(combined_df, pca_components=pca_components)
    else:
        filename = f"pv_{int(plant_id):02d}.csv"
        filepath = os.path.join(data_dir, filename)
        df = load_pv_dataset(filepath)
        X, y, pca = prepare_features_target(df, pca_components=pca_components)

    # Time series split without shuffling
    train_len = int(len(X) * (1.0 - test_size))
    X_train_raw, X_test_raw = X[:train_len], X[train_len:]
    y_train_raw, y_test_raw = y[:train_len], y[train_len:]

    if scaler_type == 'minmax':
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
    else:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train_raw)
    X_test = scaler_X.transform(X_test_raw)

    y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

    if sequence_length is not None and sequence_length > 0:
        X_train, y_train = create_sequences(X_train, y_train, time_steps=sequence_length)
        X_test, y_test = create_sequences(X_test, y_test, time_steps=sequence_length)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'pca': pca
    }

if __name__ == "__main__":
    print("Testing dataset pipeline module...")
    all_data = load_all_pv_datasets()
    print(f"Loaded {len(all_data)} PV plant datasets successfully.")

    data_ann = get_train_test_data(plant_id=1, pca_components=10)
    print(f"ANN Plant 1 X_train shape: {data_ann['X_train'].shape}, y_train shape: {data_ann['y_train'].shape}")

    data_lstm = get_train_test_data(plant_id=1, sequence_length=8, scaler_type='minmax')
    print(f"LSTM Plant 1 X_train shape: {data_lstm['X_train'].shape}, y_train shape: {data_lstm['y_train'].shape}")
