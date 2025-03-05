# Patch_TST/train_patchtst.py

import os
from pathlib import Path
import pandas as pd
import sklearn
from tsai.basics import *
from tsai.models import PatchTST
from tsai.metrics import mse, mae
from fastai.callback.tracker import EarlyStoppingCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, accuracy_score, precision_score
import numpy as np

def compute_rsi(prices, period=14):
    """
    Compute the Relative Strength Index (RSI) for a series of prices.
    Returns an array of RSI values with the same length as prices.
    The first (period-1) values will be NaN.
    """
    prices = np.array(prices)
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = np.nan
    rsi[period-1] = 100. - 100. / (1. + rs)
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi

def train_patchtst(csv_file_path, model_name='PatchTST', max_iterations=3,
                  forecast_history=24, forecast_horizon=1, batch_size=128,
                  valid_size=0.05, test_size=0.05, include_baseline=True,
                  model_history_dir=None, result_history_dir=None,
                  target='price'):
    """
    Train the PatchTST model on the given CSV file with specified parameters.

    Parameters:
        csv_file_path (str): Path to the CSV file.
        model_name (str): Name of the model (e.g., 'PatchTST').
        max_iterations (int): Number of training iterations.
        forecast_history (int): Number of past time steps to use.
        forecast_horizon (int): Number of future time steps to predict.
        batch_size (int): Batch size for training.
        valid_size (float): Fraction of data to use for validation.
        test_size (float): Fraction of data to use for testing.
        include_baseline (bool): Whether to include baseline in metrics and plots.
        model_history_dir (Path): Directory to save models for the current history.
        result_history_dir (Path): Directory to save results for the current history.
        target (str): Prediction target, either 'price' or 'rsi'.
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

    # Define model filename based on target
    if target == 'price':
        best_model_filename = f"{csv_base}_best_model.pt"
    elif target == 'rsi':
        best_model_filename = f"{csv_base}_best_model_rsi.pt"
    else:
        print("Error: Unknown target. Use 'price' or 'rsi'.")
        return
    best_model_path = model_history_dir / best_model_filename
    metrics_path = result_history_dir / f"{csv_base}_metrics.txt"

    # Lists for storing training and validation metrics
    train_mse_list = []
    val_mse_list = []
    final_learn = None  # Keep track of the last iteration's learner

    with open(metrics_path, 'w') as metrics_file:
        def log_print(*args, **kwargs):
            message = ' '.join(map(str, args))
            print(message)
            metrics_file.write(message + '\n')

        log_print(f"Model Name: {model_name} (Target: {target})")
        log_print(f"CSV Name: {csv_base}")
        log_print(f"Training Started at: {pd.Timestamp.now()}")
        log_print(f"Sequence Length (Forecast History): {forecast_history}")
        log_print(f"Forecast Horizon: {forecast_horizon}")

        # Load data
        try:
            df_raw = pd.read_csv(csv_file_path, sep=None, engine='python')
            log_print("CSV file loaded successfully.")
        except Exception as e:
            log_print(f"Error reading CSV file: {e}")
            return

        # Preprocess data
        datetime_col = "Date"
        freq = '1D'
        price_col = 'Price'
        columns = [price_col]
        method = 'ffill'
        value = 0

        preproc_pipe = sklearn.pipeline.Pipeline([
            ('shrinker', TSShrinkDataFrame()),
            ('drop_duplicates', TSDropDuplicates(datetime_col=datetime_col)),
            ('add_mts', TSAddMissingTimestamps(datetime_col=datetime_col, freq=freq)),
            ('fill_missing', TSFillMissing(columns=columns, method=method, value=value)),
        ], verbose=False)

        df = preproc_pipe.fit_transform(df_raw)
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(by=datetime_col)
        log_print("Data preprocessing completed.")

        # For RSI prediction, add the RSI column
        if target == 'rsi':
            df['RSI'] = compute_rsi(df[price_col].values, period=14)
            log_print("RSI column added to dataframe for RSI prediction.")

        # Define forecasting splits
        splits = get_forecasting_splits(
            df,
            fcst_history=forecast_history,
            fcst_horizon=forecast_horizon,
            datetime_col=datetime_col,
            valid_size=valid_size,
            test_size=test_size
        )
        log_print("Data splitting into train, validation, and test sets.")

        # Create scaling pipeline
        # (For both targets, we scale the price column)
        scale_columns = [price_col]
        exp_pipe = sklearn.pipeline.Pipeline([
            ('scaler', TSStandardScaler(columns=scale_columns)),
        ], verbose=False)

        df_scaled = exp_pipe.fit_transform(df, scaler__idxs=splits[0])
        log_print("Data scaling completed.")

        # Set the features and targets
        if target == 'price':
            x_vars = [price_col]
            y_vars = [price_col]
        elif target == 'rsi':
            x_vars = [price_col]
            y_vars = ['RSI']

        X, y = prepare_forecasting_data(
            df_scaled,
            fcst_history=forecast_history,
            fcst_horizon=forecast_horizon,
            x_vars=x_vars,
            y_vars=y_vars
        )
        log_print("Prepared forecasting data.")

        # Define model architecture configuration
        arch_config = dict(
            n_layers=3,
            n_heads=4,
            d_model=16,
            d_ff=128,
            attn_dropout=0.1,
            dropout=0.3,
            patch_len=9,
            stride=2,
            padding_patch=True
        )

        best_metric = float('inf')
        best_val_loss = float('inf')

        # Training loop
        for i in range(max_iterations):
            log_print(f"\nStarting iteration {i + 1}/{max_iterations}")

            learn = TSForecaster(
                X, y,
                splits=splits,
                batch_size=batch_size,
                path=model_history_dir,
                pipelines=[preproc_pipe, exp_pipe],
                arch=model_name,
                arch_config=arch_config,
                metrics=[mse, mae],
                cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=20)]
            )

            learn.fit_one_cycle(100, lr_max=0.001)
            log_print("Training completed for this iteration.")

            # Save the current learner (in case we need the final one)
            final_learn = learn

            val_loss = learn.recorder.values[-1][1]
            val_mse = learn.recorder.values[-1][0]
            val_mse_list.append(val_mse)

            train_mse = learn.recorder.values[-1][2] if len(learn.recorder.values[-1]) > 2 else float('nan')
            if not np.isnan(train_mse):
                train_mse_list.append(train_mse)

            try:
                scaled_preds_test, *_ = learn.get_X_preds(X[splits[2]])
                scaled_preds_test = to_np(scaled_preds_test).flatten()
                scaled_y_test = y[splits[2]].flatten()

                if target == 'price':
                    preds_test = exp_pipe.named_steps['scaler'].inverse_transform(
                        pd.DataFrame(scaled_preds_test.reshape(-1, 1), columns=scale_columns)
                    ).values.flatten()
                    actual_test = exp_pipe.named_steps['scaler'].inverse_transform(
                        pd.DataFrame(scaled_y_test.reshape(-1, 1), columns=scale_columns)
                    ).values.flatten()
                else:
                    preds_test = scaled_preds_test
                    actual_test = scaled_y_test

                test_metric = mean_squared_error(actual_test, preds_test)
                log_print(f"Iteration {i + 1}: Test MSE: {test_metric:.4f}, Validation MSE: {val_loss:.4f}")

                # (Additional logging for classification metrics, RSI error, and baseline comparison omitted for brevity)

                if test_metric < best_metric or (test_metric == best_metric and val_loss < best_val_loss):
                    best_metric = test_metric
                    best_val_loss = val_loss
                    log_print(f"Exporting best model to '{best_model_path.resolve()}'")
                    learn.export(str(best_model_path.resolve()))
                    log_print(f"New best model saved with Test Metric {best_metric:.4f} and Validation MSE {best_val_loss:.4f}")

            except Exception as e:
                log_print(f"Error during testing: {e}")

        # After all iterations, if no best model was exported, export the final iteration's model.
        if best_metric == float('inf') and final_learn is not None:
            log_print("No improvement detected across iterations; exporting final iteration model.")
            final_learn.export(str(best_model_path.resolve()))
            # Get predictions and actual values for the test set
            preds_array = to_np(final_learn.get_X_preds(X[splits[2]])[0]).reshape(-1, 1)
            preds_df = pd.DataFrame(preds_array, columns=scale_columns)
            y_array = y[splits[2]].reshape(-1, 1)
            y_df = pd.DataFrame(y_array, columns=scale_columns)
            inv_preds = exp_pipe.named_steps['scaler'].inverse_transform(preds_df)
            inv_y = exp_pipe.named_steps['scaler'].inverse_transform(y_df)
            best_metric = mean_squared_error(inv_y, inv_preds)
            log_print(f"Final model saved with Test Metric {best_metric:.4f}")

        # Log summary metrics
        if val_mse_list:
            val_mse_mean = np.mean(val_mse_list)
            val_mse_variance = np.var(val_mse_list)
            log_print(f"\nValidation MSE over {max_iterations} iterations: Mean={val_mse_mean:.4f}, Variance={val_mse_variance:.6f}")
        else:
            log_print("\nNo Validation MSE recorded.")

        if train_mse_list:
            train_mse_mean = np.mean(train_mse_list)
            train_mse_variance = np.var(train_mse_list)
            log_print(f"Training MSE over {max_iterations} iterations: Mean={train_mse_mean:.4f}, Variance={train_mse_variance:.6f}")
        else:
            log_print("No Training MSE recorded.")

        log_print(f"\nFinal Best Model for '{csv_base}' (Target: {target}) with forecast_history={forecast_history}:")
        log_print(f"  Test Metric: {best_metric:.4f}")
        log_print(f"  Validation MSE: {best_val_loss:.4f}")
        log_print(f"Model saved at: '{best_model_path}'")
