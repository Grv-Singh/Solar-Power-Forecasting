import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("Loading dataset...")
    # Load dataset with correct delimiter
    dataset = pd.read_csv('pv_01.csv', sep=';')

    # Drop index column and empty trailing column
    if 'time_idx' in dataset.columns:
        dataset = dataset.drop(columns=['time_idx'])
    if 'Unnamed: 51' in dataset.columns:
        dataset = dataset.drop(columns=['Unnamed: 51'])

    # Separate features and target
    # Target is 'power_normed'
    if 'power_normed' not in dataset.columns:
        raise ValueError("Column 'power_normed' not found in dataset.")

    X = dataset.drop(columns=['power_normed']).values
    y = dataset['power_normed'].values

    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Initialising the ANN
    model = tf.keras.models.Sequential()

    # Adding the input layer and the first hidden layer
    # Input dimension is the number of features
    model.add(tf.keras.layers.Dense(units=64, activation='relu', input_dim=X_train.shape[1]))

    # Adding the second hidden layer
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))

    # Adding a third hidden layer
    model.add(tf.keras.layers.Dense(units=16, activation='relu'))

    # Adding the output layer
    # Since target is normalized 0-1, sigmoid is a good choice for activation,
    # but linear is often safer for regression. README suggests sigmoid.
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Compiling the ANN
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the ANN to the Training set
    print("Starting training...")
    history = model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_split=0.2)

    # Predicting the Test set results
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nResults:")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R^2 Score: {r2}")

    # Save the model
    model.save('solar_ann_model.keras')
    print("Model saved to solar_ann_model.keras")

if __name__ == "__main__":
    main()
