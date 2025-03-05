# ARIMA/evaluate_arima.py

import os
from pathlib import Path
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pickle

from statsmodels.tsa.statespace.sarimax import SARIMAXResults  # To load the model

def evaluate_arima(csv_file_path, model_name='ARIMA', forecast_history=24,
                 forecast_horizon=1, batch_size=64, valid_size=0.05,
                 test_size=0.05, include_baseline=True,
                 model_history_dir=None, result_history_dir=None,
                 order=(5,1,0)):
    """
    Evaluate the trained ARIMA model on the given CSV file with specified parameters.

    Parameters:
        csv_file_path (str): Path to the CSV file.
        model_name (str): Name of the model (e.g., 'ARIMA').
        forecast_history (int): Number of past time steps to use.
        forecast_horizon (int): Number of future time steps to predict.
        batch_size (int): Batch size for evaluation (not used for ARIMA).
        valid_size (float): Fraction of data used for validation.
        test_size (float): Fraction of data used for testing.
        include_baseline (bool): Whether to include baseline in metrics and plots.
        model_history_dir (Path): Directory where the model is saved for the current history.
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
        scaler_y = MinMaxScaler(feature_range=(0, 1))

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

        # Load the trained model
        try:
            with open(model_path, 'rb') as f:
                model_fit = pickle.load(f)
            log_print(f"Model '{model_name}' loaded successfully from '{model_path}'.")
        except Exception as e:
            log_print(f"Error loading model: {e}")
            return

        # Predict on test set
        try:
            # Forecast test period
            preds_test_scaled = model_fit.predict(start=train_end, end=train_end + len(test_df) -1)
            preds_test_scaled = preds_test_scaled.reshape(-1, 1)
            preds_test = scaler_y.inverse_transform(preds_test_scaled)
            actual_test = scaler_y.inverse_transform(test_y_scaled)

            # Calculate metrics
            test_mse_model = mean_squared_error(actual_test, preds_test)
            test_mae_model = mean_absolute_error(actual_test, preds_test)
            test_rmse_model = np.sqrt(test_mse_model)

            # Baseline Predictions: Last value from training set
            last_known = scaler_y.inverse_transform(train_y_scaled[-1].reshape(1, -1))
            baseline_predictions = np.array([last_known.flatten()[0]] * len(test_df)).reshape(-1, 1)

            test_mse_baseline = mean_squared_error(actual_test, baseline_predictions)
            test_mae_baseline = mean_absolute_error(actual_test, baseline_predictions)
            test_rmse_baseline = np.sqrt(test_mse_baseline)

            # Log Test Metrics
            log_print("\nTest Metrics:")
            log_print("-------------")
            if include_baseline:
                log_print("Baseline Model:")
                log_print(f"  MAE  : {test_mae_baseline:.6f}")
                log_print(f"  MSE  : {test_mse_baseline:.6f}")
                log_print(f"  RMSE : {test_rmse_baseline:.6f}\n")
            log_print(f"{model_name} Model:")
            log_print(f"  MAE  : {test_mae_model:.6f}")
            log_print(f"  MSE  : {test_mse_model:.6f}")
            log_print(f"  RMSE : {test_rmse_model:.6f}\n")

            if include_baseline:
                # Compare model and baseline in terms of MSE
                mse_diff = test_mse_model - test_mse_baseline
                mse_percentage = (mse_diff / test_mse_baseline) * 100

                if mse_percentage > 0:
                    log_print(f"Model MSE is worse by {mse_percentage:.2f}% compared to the baseline.")
                else:
                    log_print(f"Model MSE is better by {abs(mse_percentage):.2f}% compared to the baseline.")

        except Exception as e:
            log_print(f"Error during testing: {e}")

        # Prepare data for plotting
        try:
            indices = np.arange(len(actual_test))

            # Create the plot
            plt.figure(figsize=(14, 7))

            # Plot Actual Prices
            plt.plot(indices, actual_test, label='Actual Price', marker='o', linestyle='-', color='blue')

            # Plot Predicted Prices from ARIMA Model
            plt.plot(indices, preds_test, label=f'{model_name} Predictions', marker='x', linestyle='--', color='red')

            if include_baseline:
                # Plot Baseline Predictions
                plt.plot(indices, baseline_predictions, label='Baseline Prediction', marker='s', linestyle=':', color='green')

            # Enhancements
            plt.xlabel('Sample Index', fontsize=14)
            plt.ylabel('Price', fontsize=14)
            plt.title(f'Actual vs. Predicted{" vs. Baseline" if include_baseline else ""} Prices (Test Set)',
                      fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()

            # Save the plot with forecast_history in the filename
            plot_filename = f"{csv_base}_{forecast_history}_{forecast_horizon}_test_plot.png"
            plot_path = result_history_dir / plot_filename
            plt.savefig(plot_path)
            plt.close()
            log_print(f"Plot saved to '{plot_path}'")

        except Exception as e:
            log_print(f"Error during plotting: {e}")

        # Log final evaluation info
        log_print(
            f"\nEvaluation completed for '{csv_base}' with model '{model_name}' and forecast_history={forecast_history}.")
        log_print(f"Metrics and plot appended to '{metrics_path}' and saved plot at '{plot_path}'")
