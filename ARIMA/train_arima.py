# ARIMA/train_arima.py

import os
from pathlib import Path
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from tqdm import tqdm
import pickle

from statsmodels.tsa.statespace.sarimax import SARIMAX  # Use SARIMAX for ARIMA with exogenous variables

def train_arima(csv_file_path, model_name='ARIMA', max_iterations=1,
               forecast_history=24, forecast_horizon=1, batch_size=64,
               valid_size=0.05, test_size=0.05, include_baseline=True,
               model_history_dir=None, result_history_dir=None,
               order=(5,1,0)):
    """
    Train the ARIMA model on the given CSV file with specified parameters.

    Parameters:
        csv_file_path (str): Path to the CSV file.
        model_name (str): Name of the model (e.g., 'ARIMA').
        max_iterations (int): Number of training iterations.
        forecast_history (int): Number of past time steps to use.
        forecast_horizon (int): Number of future time steps to predict.
        batch_size (int): Batch size for training (not used for ARIMA).
        valid_size (float): Fraction of data to use for validation.
        test_size (float): Fraction of data to use for testing.
        include_baseline (bool): Whether to include baseline in metrics and plots.
        model_history_dir (Path): Directory to save models for the current history.
        result_history_dir (Path): Directory to save results for the current history.
        order (tuple): ARIMA order (p, d, q).
    """
    # Ensure the CSV file exists
    if not os.path.isfile(csv_file_path):
        print(f"Error: The file '{csv_file_path}' does not exist.")
        return

    # Extract base name without extension for naming
    csv_base = Path(csv_file_path).stem

    # Validate directory parameters
    if model_history_dir is None or result_history_dir is None:
        print("Error: 'model_history_dir' and 'result_history_dir' must be provided.")
        return

    # Define paths for saving the best model and its metrics
    best_model_filename = f"{csv_base}_best_model.pkl"  # Use <csv>_best_model.pkl
    best_model_path = model_history_dir / best_model_filename
    metrics_path = result_history_dir / f"{csv_base}_metrics.txt"

    # Initialize variables to track the best model
    best_mse = float('inf')

    # Initialize scalers
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    # Open a text file to write validation and test metrics
    with open(metrics_path, 'w') as metrics_file:
        def log_print(*args, **kwargs):
            """
            Helper function to print messages and write to the metrics file.
            """
            message = ' '.join(map(str, args))
            print(message)
            metrics_file.write(message + '\n')

        # Log model and CSV info
        log_print(f"Model Name: {model_name}")
        log_print(f"CSV Name: {csv_base}")
        log_print(f"Training Started at: {pd.Timestamp.now()}")
        log_print(f"Sequence Length (Forecast History): {forecast_history}")
        log_print(f"Forecast Horizon: {forecast_horizon}")
        log_print(f"ARIMA Order: {order}")

        # Load data
        try:
            df_raw = pd.read_csv(csv_file_path)
            log_print("CSV file loaded successfully.")
        except Exception as e:
            log_print(f"Error reading CSV file: {e}")
            return

        # Preprocess data
        datetime_col = "Date"
        freq = '1D'
        columns = ['Price']
        method = 'ffill'
        value = 0

        # Preprocessing steps: drop duplicates, fill missing, sort
        preproc_pipe = sklearn.pipeline.Pipeline([
            ('drop_duplicates', sklearn.preprocessing.FunctionTransformer(lambda df: df.drop_duplicates(subset=datetime_col))),
            ('fill_missing', sklearn.preprocessing.FunctionTransformer(lambda df: df.fillna(method=method).fillna(value=value)))
        ])

        df = preproc_pipe.fit_transform(df_raw)
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(by=datetime_col)
        log_print("Data preprocessing completed.")

        # Define forecasting splits
        total_size = len(df)
        test_split = int(total_size * test_size)
        valid_split = int(total_size * valid_size)

        train_end = total_size - test_split - valid_split
        valid_end = total_size - test_split

        train_df = df.iloc[:train_end]
        valid_df = df.iloc[train_end:valid_end]
        test_df = df.iloc[valid_end:]

        log_print(f"Data split into train ({len(train_df)}), validation ({len(valid_df)}), and test ({len(test_df)}) sets.")

        # Scale the data
        train_y = train_df[['Price']].values
        scaler_y.fit(train_y)

        train_y_scaled = scaler_y.transform(train_y)
        valid_y = valid_df[['Price']].values
        valid_y_scaled = scaler_y.transform(valid_y)
        test_y = test_df[['Price']].values
        test_y_scaled = scaler_y.transform(test_y)

        log_print("Data scaling completed.")

        # Prepare forecasting data
        # ARIMA is a univariate model; exog should be None or meaningful exogenous variables
        # Here, we proceed without exogenous variables

        for i in range(max_iterations):
            log_print(f"\nStarting training iteration {i + 1}/{max_iterations}")

            # Fit ARIMA on training data
            try:
                model = SARIMAX(train_y_scaled,
                                order=order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                model_fit = model.fit(disp=False)
                log_print(f"ARIMA model fitted: order={order}")
            except Exception as e:
                log_print(f"Error fitting ARIMA model: {e}")
                continue

            # Predict on validation set
            try:
                # Forecast validation period
                preds_valid_scaled = model_fit.predict(start=train_end, end=valid_end -1)
                preds_valid_scaled = preds_valid_scaled.reshape(-1, 1)
                preds_valid = scaler_y.inverse_transform(preds_valid_scaled)
                actual_valid = scaler_y.inverse_transform(valid_y_scaled)

                # Calculate validation MSE
                val_mse = mean_squared_error(actual_valid, preds_valid)
                log_print(f"Iteration {i + 1}: Validation MSE: {val_mse:.6f}")

                # Check if this is the best model so far
                if val_mse < best_mse:
                    best_mse = val_mse
                    # Save the model
                    with open(best_model_path, 'wb') as f:
                        pickle.dump(model_fit, f)
                    log_print(f"New best model saved at '{best_model_path}' with Validation MSE {best_mse:.6f}")
            except Exception as e:
                log_print(f"Error during validation prediction: {e}")

        # After training iterations, log the best model info
        log_print(f"\nTraining completed. Best Validation MSE: {best_mse:.6f}")
        log_print(f"Best model saved at: '{best_model_path}'")

        # Note: Test evaluation is handled in the evaluation script

        log_print(f"Metrics saved at '{metrics_path}'")