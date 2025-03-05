# Prophet/train_prophet.py

import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle
from prophet import Prophet
import numpy as np

class MyProphet:
    def __init__(self, response_col='y', date_col='ds'):
        self.response_col = response_col
        self.date_col = date_col
        self.model = Prophet()

    def add_regressors(self, regressors):
        for reg in regressors:
            self.model.add_regressor(reg)

    def fit(self, data_x):
        self.model.fit(data_x)

    def predict(self, data_x):
        forecast = self.model.predict(data_x)
        return forecast  # Return the entire forecast DataFrame

def train_prophet(csv_file_path, model_name='Prophet', max_iterations=1,
                 forecast_history=24, forecast_horizon=1, batch_size=64,
                 valid_size=0.05, test_size=0.05, include_baseline=True,
                 model_history_dir=None, result_history_dir=None):
    """
    Train the Prophet model on the given CSV file with specified parameters.

    Parameters:
        csv_file_path (str): Path to the CSV file.
        model_name (str): Name of the model (e.g., 'Prophet').
        max_iterations (int): Number of training iterations.
        forecast_history (int): Number of past time steps to use.
        forecast_horizon (int): Number of future time steps to predict.
        batch_size (int): Batch size for training (not applicable for Prophet but kept for consistency).
        valid_size (float): Fraction of data to use for validation.
        test_size (float): Fraction of data to use for testing.
        include_baseline (bool): Whether to include baseline in metrics and plots.
        model_history_dir (Path): Directory to save models for the current history.
        result_history_dir (Path): Directory to save results for the current history.
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

    # Initialize variable to track the best model
    best_mse = float('inf')

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

        # Load data
        try:
            df_raw = pd.read_csv(csv_file_path)
            log_print("CSV file loaded successfully.")
        except Exception as e:
            log_print(f"Error reading CSV file: {e}")
            return

        # Preprocess data
        datetime_col = "Date"
        price_col = "Price"

        # Ensure datetime is in correct format
        df_raw[datetime_col] = pd.to_datetime(df_raw[datetime_col])
        df_raw = df_raw.sort_values(by=datetime_col).reset_index(drop=True)
        log_print("Data preprocessing completed.")

        # Split data into train, validation, and test sets
        total_size = len(df_raw)
        test_split = int(total_size * test_size)
        valid_split = int(total_size * valid_size)

        train_end = total_size - test_split - valid_split
        valid_end = total_size - test_split

        # **Important Fix:** Use .copy() to avoid SettingWithCopyError
        train_df = df_raw.iloc[:train_end].copy()
        valid_df = df_raw.iloc[train_end:valid_end].copy()
        test_df = df_raw.iloc[valid_end:].copy()

        log_print(f"Data split into train ({len(train_df)}), validation ({len(valid_df)}), and test ({len(test_df)}) sets.")

        # **No Scaling of 'y'**
        # If you have additional regressors, scale them here
        # For example:
        # scaler = StandardScaler()
        # train_df['regressor'] = scaler.fit_transform(train_df[['regressor']])
        # valid_df['regressor'] = scaler.transform(valid_df[['regressor']])
        # test_df['regressor'] = scaler.transform(test_df[['regressor']])

        log_print("Data scaling (excluding 'y') completed.")

        for iteration in range(max_iterations):
            log_print(f"\nStarting training iteration {iteration + 1}/{max_iterations}")

            # Initialize the Prophet model
            prophet_model = MyProphet(response_col=price_col, date_col=datetime_col)

            # If you have additional regressors, add them here
            # For example:
            # regressors = ['feature1', 'feature2']
            # prophet_model.add_regressors(regressors)

            # Prepare training data
            train_prophet_df = train_df.rename(columns={datetime_col: 'ds', price_col: 'y'})

            # Initialize and fit the model
            try:
                prophet_model.fit(train_prophet_df)
                log_print("Model training completed.")
            except Exception as e:
                log_print(f"Error during model training: {e}")
                continue

            # Validate the model on validation set
            valid_prophet_df = valid_df.rename(columns={datetime_col: 'ds', price_col: 'y'})
            try:
                preds_valid = prophet_model.predict(valid_prophet_df)
                mse_val = mean_squared_error(valid_prophet_df['y'], preds_valid['yhat'])
                log_print(f"Iteration {iteration + 1}: Validation MSE: {mse_val:.6f}")
            except Exception as e:
                log_print(f"Error during validation: {e}")
                continue

            # Check if this model is better than the previous best
            if mse_val < best_mse:
                best_mse = mse_val
                # Save the best model
                try:
                    with open(best_model_path, 'wb') as f:
                        pickle.dump(prophet_model, f)
                    log_print(f"New best model saved at '{best_model_path}' with Validation MSE {best_mse:.6f}")
                except Exception as e:
                    log_print(f"Error saving the model: {e}")