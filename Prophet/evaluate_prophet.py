# Prophet/evaluate_prophet.py

import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet

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

def evaluate_prophet(csv_file_path, model_name='Prophet', forecast_history=24,
                    forecast_horizon=1, batch_size=64, valid_size=0.05,
                    test_size=0.05, include_baseline=True,
                    model_history_dir=None, result_history_dir=None):
    """
    Evaluate the trained Prophet model on the given CSV file with specified parameters.

    Parameters:
        csv_file_path (str): Path to the CSV file.
        model_name (str): Name of the model (e.g., 'Prophet').
        forecast_history (int): Number of past time steps to use.
        forecast_horizon (int): Number of future time steps to predict.
        batch_size (int): Batch size for evaluation (not applicable for Prophet but kept for consistency).
        valid_size (float): Fraction of data used for validation.
        test_size (float): Fraction of data used for testing.
        include_baseline (bool): Whether to include baseline in metrics and plots.
        model_history_dir (Path): Directory where the model is saved for the current history.
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

    # Define path for the metrics file (same as training)
    metrics_path = result_history_dir / f"{csv_base}_metrics.txt"

    # Define model filename
    model_filename = f"{csv_base}_best_model.pkl"  # Should match the training script
    model_path = model_history_dir / model_filename

    # Check if the model exists
    if not model_path.exists():
        print(f"Error: The model '{model_path}' does not exist.")
        return

    # Open the metrics file in append mode
    with open(metrics_path, 'a') as metrics_file:
        def log_print(*args, **kwargs):
            """
            Helper function to print messages and write to the metrics file.
            """
            message = ' '.join(map(str, args))
            print(message)
            metrics_file.write(message + '\n')

        # Log evaluation info
        log_print(f"\n--- Evaluation at {pd.Timestamp.now()} ---")
        log_print(f"Model Name: {model_name}")
        log_print(f"CSV Name: {csv_base}")
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

        # Load the trained model
        try:
            with open(model_path, 'rb') as f:
                prophet_model = pickle.load(f)
            log_print(f"Model '{model_name}' loaded successfully from '{model_path}'.")
        except Exception as e:
            log_print(f"Error loading the model: {e}")
            return

        # Prepare test data for Prophet
        test_prophet_df = test_df.rename(columns={datetime_col: 'ds', price_col: 'y'})

        # Log the columns of test_prophet_df for debugging
        log_print(f"Columns in test_prophet_df: {test_prophet_df.columns.tolist()}")
        log_print(f"Sample data in test_prophet_df:\n{test_prophet_df.head()}")

        # Make predictions
        try:
            preds_test = prophet_model.predict(test_prophet_df)
            # preds_test is a DataFrame with a 'yhat' column
            # No need to rename 'yhat' as it's already correctly named
            preds_test = preds_test[['ds', 'yhat']]
            # Since 'y' is not scaled, 'yhat' is also in original scale
            actuals = test_prophet_df['y'].values
            predictions = preds_test['yhat'].values
        except Exception as e:
            log_print(f"Error during prediction: {e}")
            return

        # Calculate evaluation metrics
        try:
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mse)

            log_print("\nTest Metrics:")
            log_print("-------------")
            log_print(f"{model_name} Model:")
            log_print(f"  MAE  : {mae:.6f}")
            log_print(f"  MSE  : {mse:.6f}")
            log_print(f"  RMSE : {rmse:.6f}\n")
        except Exception as e:
            log_print(f"Error calculating metrics: {e}")
            return

        if include_baseline:
            # Baseline Prediction: Last value of the input sequence
            # Here, assuming the last observed price is the baseline
            # Adjust this as per your actual baseline strategy
            baseline_predictions = test_df[price_col].values  # Directly access 'Price' column

            try:
                baseline_mse = mean_squared_error(actuals, baseline_predictions)
                baseline_mae = mean_absolute_error(actuals, baseline_predictions)
                baseline_rmse = np.sqrt(baseline_mse)

                log_print("Baseline Metrics:")
                log_print(f"  MAE  : {baseline_mae:.6f}")
                log_print(f"  MSE  : {baseline_mse:.6f}")
                log_print(f"  RMSE : {baseline_rmse:.6f}\n")

                # Compare model and baseline in terms of MSE
                mse_diff = mse - baseline_mse
                mse_percentage = (mse_diff / baseline_mse) * 100

                if mse_percentage > 0:
                    log_print(f"Model MSE is worse by {mse_percentage:.2f}% compared to the baseline.")
                else:
                    log_print(f"Model MSE is better by {abs(mse_percentage):.2f}% compared to the baseline.")
            except Exception as e:
                log_print(f"Error during baseline evaluation: {e}")

        # Log final evaluation info
        log_print(f"\nEvaluation completed for '{csv_base}' with model '{model_name}' and forecast_history={forecast_history}.")
