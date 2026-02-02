import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import data_loader
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description='Train ANN model for Solar Power Forecasting')
    parser.add_argument('--file', type=str, default='pv_01.csv', help='Path to the CSV dataset file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    # Load and preprocess
    try:
        dataset = data_loader.load_data(args.file)
        X_train, X_test, y_train, y_test, scaler = data_loader.preprocess_data_ann(dataset)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Initialising the ANN
    model = tf.keras.models.Sequential()

    # Adding the input layer and the first hidden layer
    model.add(tf.keras.layers.Dense(units=64, activation='relu', input_dim=X_train.shape[1]))

    # Adding the second hidden layer
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))

    # Adding a third hidden layer
    model.add(tf.keras.layers.Dense(units=16, activation='relu'))

    # Adding the output layer
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Compiling the ANN
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the ANN to the Training set
    print(f"Starting training on {args.file} for {args.epochs} epochs...")
    history = model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1, validation_split=0.2)

    # Predicting the Test set results
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nResults for {args.file}:")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R^2 Score: {r2}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:100], label='Actual')
    plt.plot(y_pred[:100], label='Predicted')
    plt.title(f'ANN: Actual vs Predicted Power (First 100 samples) - {args.file}')
    plt.xlabel('Sample')
    plt.ylabel('Normalized Power')
    plt.legend()
    plot_filename = f'ann_results_{os.path.basename(args.file).split(".")[0]}.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

if __name__ == "__main__":
    main()
