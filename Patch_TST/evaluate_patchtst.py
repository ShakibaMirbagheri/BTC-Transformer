# Patch_TST/evaluate_patchtst.py

import os
from pathlib import Path
import pandas as pd
import sklearn
from tsai.basics import *
from tsai.models import PatchTST
from tsai.metrics import mse, mae
from sklearn.metrics import (mean_squared_error, mean_absolute_error, accuracy_score,
                             f1_score, precision_score, r2_score, mean_absolute_percentage_error)
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import mplfinance as mpf

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

def compute_macd(prices, fast=12, slow=26, signal=9):
    """
    Compute the MACD (Moving Average Convergence Divergence) indicator.
    Returns MACD line, Signal line and Histogram.
    """
    prices = pd.Series(prices)
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd.values, signal_line.values, histogram.values

def evaluate_patchtst(csv_file_path, model_name='PatchTST', forecast_history=24,
                      forecast_horizon=1, batch_size=128, valid_size=0.05,
                      test_size=0.05, include_baseline=True,
                      model_history_dir=None, result_history_dir=None,
                      target='price'):
    """
    Evaluate the trained PatchTST model on the given CSV file with specified parameters.

    Parameters:
        csv_file_path (str): Path to the CSV file.
        model_name (str): Name of the model (e.g., 'PatchTST').
        forecast_history (int): Number of past time steps to use.
        forecast_horizon (int): Number of future time steps to predict.
        batch_size (int): Batch size for evaluation.
        valid_size (float): Fraction of data used for validation.
        test_size (float): Fraction of data used for testing.
        include_baseline (bool): Whether to include baseline in metrics and plots.
        model_history_dir (Path): Directory where the model is saved for the current history.
        result_history_dir (Path): Directory to save results for the current history.
        target (str): Prediction target, either 'price' or 'rsi'.
    """
    # Ensure the CSV file exists
    if not os.path.isfile(csv_file_path):
        print(f"Error: The file '{csv_file_path}' does not exist.")
        return

    csv_base = Path(csv_file_path).stem

    if model_history_dir is None or result_history_dir is None:
        print("Error: 'model_history_dir' and 'result_history_dir' must be provided.")
        return

    # Metrics file path (append evaluation logs)
    metrics_path = result_history_dir / f"{csv_base}_metrics.txt"

    # Define model filename based on target
    if target == 'price':
        model_filename = f"{csv_base}_best_model.pt"
    elif target == 'rsi':
        model_filename = f"{csv_base}_best_model_rsi.pt"
    else:
        print("Error: Unknown target. Use 'price' or 'rsi'.")
        return
    model_path = model_history_dir / model_filename

    if not model_path.exists():
        print(f"Error: The model '{model_path}' does not exist.")
        return

    with open(metrics_path, 'a') as metrics_file:
        def log_print(*args, **kwargs):
            message = ' '.join(map(str, args))
            print(message)
            metrics_file.write(message + '\n')

        log_print(f"\n--- Evaluation at {pd.Timestamp.now()} ---")
        log_print(f"Model Name: {model_name} (Target: {target})")
        log_print(f"CSV Name: {csv_base}")
        log_print(f"Forecast History: {forecast_history}")
        log_print(f"Forecast Horizon: {forecast_horizon}")

        # Load data
        try:
            df_raw = pd.read_csv(csv_file_path, sep=None, engine='python')
            log_print("CSV file loaded successfully.")
        except Exception as e:
            log_print(f"Error reading CSV file: {e}")
            return

        # Preprocess data (using only the Price column for scaling and forecasting)
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

        # If target is 'rsi', add RSI column
        if target == 'rsi':
            df['RSI'] = compute_rsi(df[price_col].values, period=14)
            log_print("RSI column added to dataframe for RSI evaluation.")

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

        # For evaluation, use MinMaxScaler on price column
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
        scale_columns = [price_col]
        df_scaled = df.copy()
        df_scaled[scale_columns] = scaler.fit_transform(df[scale_columns])
        log_print("Data scaling completed using MinMaxScaler.")

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

        # Load the trained model
        try:
            learn = load_learner(str(model_path.resolve()))
            log_print(f"Model '{model_name}' loaded successfully.")
        except Exception as e:
            log_print(f"Error loading model: {e}")
            return

        # -------------------------
        # TEST SET EVALUATION
        # -------------------------
        try:
            scaled_preds_test, *_ = learn.get_X_preds(X[splits[2]])
            scaled_preds_test = to_np(scaled_preds_test).flatten()
            scaled_y_test = y[splits[2]].flatten()

            if target == 'price':
                preds_test = scaler.inverse_transform(
                    pd.DataFrame(scaled_preds_test.reshape(-1, 1), columns=scale_columns)
                ).flatten()
                actual_test = scaler.inverse_transform(
                    pd.DataFrame(scaled_y_test.reshape(-1, 1), columns=scale_columns)
                ).flatten()
            else:
                preds_test = scaled_preds_test
                actual_test = scaled_y_test

            test_mse_model = mean_squared_error(actual_test, preds_test)
            test_mae_model = mean_absolute_error(actual_test, preds_test)
            test_rmse_model = np.sqrt(test_mse_model)
            test_r2 = r2_score(actual_test, preds_test)
            test_mape = mean_absolute_percentage_error(actual_test, preds_test)

            log_print("\nTest Regression Metrics:")
            if include_baseline and target == 'price':
                y_baseline_test = scaler.inverse_transform(
                    pd.DataFrame(X[splits[2]][:, 0, -1].reshape(-1, 1), columns=scale_columns)
                ).flatten()
                baseline_mse = mean_squared_error(actual_test, y_baseline_test)
                log_print("Baseline Model:")
                log_print(f"  MAE  : {mean_absolute_error(actual_test, y_baseline_test):.4f}")
                log_print(f"  MSE  : {baseline_mse:.4f}")
                log_print(f"  RMSE : {np.sqrt(baseline_mse):.4f}\n")
            log_print(f"{model_name} Model:")
            log_print(f"  MAE  : {test_mae_model:.4f}")
            log_print(f"  MSE  : {test_mse_model:.4f}")
            log_print(f"  RMSE : {test_rmse_model:.4f}")
            log_print(f"  R2   : {test_r2:.4f}")
            log_print(f"  MAPE : {test_mape:.4f}")

            if target == 'price':
                previous_prices = scaler.inverse_transform(
                    pd.DataFrame(X[splits[2]][:, 0, -1].reshape(-1, 1), columns=scale_columns)
                ).flatten()
                actual_directions = (actual_test > previous_prices).astype(int)
                predicted_directions = (preds_test > previous_prices).astype(int)
                acc = accuracy_score(actual_directions, predicted_directions)
                f1 = f1_score(actual_directions, predicted_directions)
                prec = precision_score(actual_directions, predicted_directions)
                log_print("\nTest Classification Metrics:")
                log_print(f"  Accuracy : {acc:.4f}")
                log_print(f"  F1-Score : {f1:.4f}")
                log_print(f"  Precision: {prec:.4f}")

                # RSI error metrics
                if len(actual_test) >= 14:
                    rsi_actual = compute_rsi(actual_test, period=14)
                    rsi_pred = compute_rsi(preds_test, period=14)
                    valid_idx = ~np.isnan(rsi_actual)
                    if valid_idx.sum() > 0:
                        rsi_mae = mean_absolute_error(rsi_actual[valid_idx], rsi_pred[valid_idx])
                        rsi_mse = mean_squared_error(rsi_actual[valid_idx], rsi_pred[valid_idx])
                        log_print(f"\nRSI Metrics:")
                        log_print(f"  RSI MAE: {rsi_mae:.4f}")
                        log_print(f"  RSI MSE: {rsi_mse:.4f}")

                # MACD error metrics (if enough data)
                if len(actual_test) >= 26:
                    macd_actual, _, _ = compute_macd(actual_test)
                    macd_pred, _, _ = compute_macd(preds_test)
                    macd_mae = mean_absolute_error(macd_actual, macd_pred)
                    log_print(f"\nMACD Metric:")
                    log_print(f"  MACD MAE: {macd_mae:.4f}")

            # Plotting test set predictions: Prices and RSI
            indices = np.arange(len(actual_test))
            plt.figure(figsize=(14, 10))
            if target == 'price':
                plt.subplot(2, 1, 1)
                plt.plot(indices, actual_test, label='Actual Price', marker='o', linestyle='-', color='blue')
                plt.plot(indices, preds_test, label='Predicted Price', marker='x', linestyle='--', color='red')
                if include_baseline:
                    plt.plot(indices, y_baseline_test, label='Baseline Prediction', marker='s', linestyle=':', color='green')
                plt.xlabel('Sample Index', fontsize=14)
                plt.ylabel('Price', fontsize=14)
                plt.title(f"Test Set - {model_name}: Actual vs Predicted Prices", fontsize=16)
                plt.legend(fontsize=12)
                plt.grid(True)

                plt.subplot(2, 1, 2)
                if len(actual_test) >= 14:
                    rsi_actual = compute_rsi(actual_test, period=14)
                    rsi_pred = compute_rsi(preds_test, period=14)
                    plt.plot(indices, rsi_actual, label='Actual RSI', marker='o', linestyle='-', color='blue')
                    plt.plot(indices, rsi_pred, label='Predicted RSI', marker='x', linestyle='--', color='red')
                    plt.xlabel('Sample Index', fontsize=14)
                    plt.ylabel('RSI', fontsize=14)
                    plt.title(f"Test Set - {model_name}: RSI Comparison", fontsize=16)
                    plt.legend(fontsize=12)
                    plt.grid(True)
                else:
                    plt.text(0.5, 0.5, 'Not enough data for RSI calculation', fontsize=14, ha='center')
                plt.tight_layout()
                test_plot_filename = f"{csv_base}_{forecast_history}_{forecast_horizon}_test_plot.png"
                test_plot_path = result_history_dir / test_plot_filename
                plt.savefig(test_plot_path)
                plt.close()
                log_print(f"Test prediction plot saved to '{test_plot_path}'")

            # Create candlestick chart with RSI for the test period (using full data)
            try:
                df_candle = df_raw.copy()
                df_candle[datetime_col] = pd.to_datetime(df_candle[datetime_col])
                df_candle = df_candle.sort_values(by=datetime_col)
                if 'Price' in df_candle.columns:
                    df_candle.rename(columns={'Price': 'Close'}, inplace=True)
                # Check if volume data exists in your DataFrame. If the column is named "Vol.", rename it to "Volume"
                if 'Vol.' in df_candle.columns:
                    df_candle.rename(columns={'Vol.': 'Volume'}, inplace=True)
                    df_candle['Volume'] = df_candle['Volume'].apply(lambda x: float(x.replace('K',''))*1000 if isinstance(x, str) and 'K' in x else float(x))
                df_candle.set_index(datetime_col, inplace=True)
                rsi_values = compute_rsi(df_candle['Close'].values, period=14)
                df_candle['RSI'] = rsi_values
                apds = [mpf.make_addplot(df_candle['RSI'], panel=1, color='blue', ylabel='RSI')]
                mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
                s  = mpf.make_mpf_style(marketcolors=mc)
                # Plot volume if available; otherwise, volume=False
                volume_flag = True if 'Volume' in df_candle.columns else False
                fig, axlist = mpf.plot(df_candle, type='candle', style=s, volume=volume_flag,
                                       addplot=apds, returnfig=True, figsize=(14,10), panel_ratios=(3,1))
                fig.suptitle(f"Test Set - {model_name}: Candlestick Chart with RSI", fontsize=16)
                candle_plot_filename = f"{csv_base}_{forecast_history}_{forecast_horizon}_candlestick.png"
                candle_plot_path = result_history_dir / candle_plot_filename
                fig.savefig(candle_plot_path)
                plt.close(fig)
                log_print(f"Candlestick chart with RSI saved to '{candle_plot_path}'")
            except Exception as e:
                log_print(f"Error during test candlestick plotting: {e}")

        except Exception as e:
            log_print(f"Error during test evaluation: {e}")

        # -------------------------
        # VALIDATION SET EVALUATION
        # -------------------------
        try:
            scaled_preds_val, *_ = learn.get_X_preds(X[splits[1]])
            scaled_preds_val = to_np(scaled_preds_val).flatten()
            scaled_y_val = y[splits[1]].flatten()

            if target == 'price':
                preds_val = scaler.inverse_transform(
                    pd.DataFrame(scaled_preds_val.reshape(-1, 1), columns=scale_columns)
                ).flatten()
                actual_val = scaler.inverse_transform(
                    pd.DataFrame(scaled_y_val.reshape(-1, 1), columns=scale_columns)
                ).flatten()
            else:
                preds_val = scaled_preds_val
                actual_val = scaled_y_val

            val_mse = mean_squared_error(actual_val, preds_val)
            val_mae = mean_absolute_error(actual_val, preds_val)
            val_rmse = np.sqrt(val_mse)
            val_r2 = r2_score(actual_val, preds_val)
            val_mape = mean_absolute_percentage_error(actual_val, preds_val)

            log_print("\nValidation Regression Metrics:")
            log_print(f"  MAE  : {val_mae:.4f}")
            log_print(f"  MSE  : {val_mse:.4f}")
            log_print(f"  RMSE : {val_rmse:.4f}")
            log_print(f"  R2   : {val_r2:.4f}")
            log_print(f"  MAPE : {val_mape:.4f}")

            if target == 'price':
                y_baseline_val = scaler.inverse_transform(
                    pd.DataFrame(X[splits[1]][:, 0, -1].reshape(-1, 1), columns=scale_columns)
                ).flatten()
                baseline_val_mse = mean_squared_error(actual_val, y_baseline_val)
                log_print(f"  Baseline MSE : {baseline_val_mse:.4f}")
                previous_prices_val = scaler.inverse_transform(
                    pd.DataFrame(X[splits[1]][:, 0, -1].reshape(-1, 1), columns=scale_columns)
                ).flatten()
                actual_dir_val = (actual_val > previous_prices_val).astype(int)
                pred_dir_val = (preds_val > previous_prices_val).astype(int)
                val_acc = accuracy_score(actual_dir_val, pred_dir_val)
                val_f1 = f1_score(actual_dir_val, pred_dir_val)
                val_prec = precision_score(actual_dir_val, pred_dir_val)
                log_print("  Direction Metrics:")
                log_print(f"    Accuracy : {val_acc:.4f}")
                log_print(f"    F1-Score : {val_f1:.4f}")
                log_print(f"    Precision: {val_prec:.4f}")

                if len(actual_val) >= 14:
                    rsi_actual_val = compute_rsi(actual_val, period=14)
                    rsi_pred_val = compute_rsi(preds_val, period=14)
                    valid_idx = ~np.isnan(rsi_actual_val)
                    if valid_idx.sum() > 0:
                        rsi_mae_val = mean_absolute_error(rsi_actual_val[valid_idx], rsi_pred_val[valid_idx])
                        rsi_mse_val = mean_squared_error(rsi_actual_val[valid_idx], rsi_pred_val[valid_idx])
                        log_print("  RSI Metrics:")
                        log_print(f"    RSI MAE: {rsi_mae_val:.4f}")
                        log_print(f"    RSI MSE: {rsi_mse_val:.4f}")

                if len(actual_val) >= 26:
                    macd_actual_val, _, _ = compute_macd(actual_val)
                    macd_pred_val, _, _ = compute_macd(preds_val)
                    macd_mae_val = mean_absolute_error(macd_actual_val, macd_pred_val)
                    log_print("  MACD Metric:")
                    log_print(f"    MACD MAE: {macd_mae_val:.4f}")

            # Plot validation predictions: Prices and RSI
            indices_val = np.arange(len(actual_val))
            plt.figure(figsize=(14,10))
            if target == 'price':
                plt.subplot(2,1,1)
                plt.plot(indices_val, actual_val, label='Actual Price', marker='o', linestyle='-', color='blue')
                plt.plot(indices_val, preds_val, label='Predicted Price', marker='x', linestyle='--', color='red')
                plt.xlabel('Sample Index', fontsize=14)
                plt.ylabel('Price', fontsize=14)
                plt.title(f"Validation Set - {model_name}: Actual vs Predicted Prices", fontsize=16)
                plt.legend(fontsize=12)
                plt.grid(True)

                plt.subplot(2,1,2)
                if len(actual_val) >= 14:
                    rsi_actual_val = compute_rsi(actual_val, period=14)
                    rsi_pred_val = compute_rsi(preds_val, period=14)
                    plt.plot(indices_val, rsi_actual_val, label='Actual RSI', marker='o', linestyle='-', color='blue')
                    plt.plot(indices_val, rsi_pred_val, label='Predicted RSI', marker='x', linestyle='--', color='red')
                    plt.xlabel('Sample Index', fontsize=14)
                    plt.ylabel('RSI', fontsize=14)
                    plt.title(f"Validation Set - {model_name}: RSI Comparison", fontsize=16)
                    plt.legend(fontsize=12)
                    plt.grid(True)
                plt.tight_layout()
                val_plot_filename = f"{csv_base}_{forecast_history}_{forecast_horizon}_val_plot.png"
                val_plot_path = result_history_dir / val_plot_filename
                plt.savefig(val_plot_path)
                plt.close()
                log_print(f"Validation prediction plot saved to '{val_plot_path}'")

                # Create a candlestick chart for the validation period.
                try:
                    df_candle_val = df_raw.copy()
                    df_candle_val[datetime_col] = pd.to_datetime(df_candle_val[datetime_col])
                    df_candle_val = df_candle_val.sort_values(by=datetime_col)
                    # Approximate the validation period as the last N rows (where N = len(splits[1]))
                    val_len = len(splits[1])
                    df_val = df_candle_val.iloc[-val_len:].copy()  # Explicit copy to avoid SettingWithCopy warning
                    if 'Price' in df_val.columns:
                        df_val.rename(columns={'Price': 'Close'}, inplace=True)
                    # Check for volume column: if "Vol." exists, rename it to "Volume" and convert to numeric
                    if 'Vol.' in df_val.columns:
                        df_val.rename(columns={'Vol.': 'Volume'}, inplace=True)
                        df_val['Volume'] = df_val['Volume'].apply(lambda x: float(x.replace('K',''))*1000 if isinstance(x, str) and 'K' in x else float(x))
                    df_val.set_index(datetime_col, inplace=True)
                    rsi_val = compute_rsi(df_val['Close'].values, period=14)
                    df_val['RSI'] = rsi_val
                    # Overlay predicted prices as a new column.
                    df_val['Predicted'] = preds_val[:len(df_val)]
                    apds_val = [
                        mpf.make_addplot(df_val['RSI'], panel=1, color='blue', ylabel='RSI'),
                        mpf.make_addplot(df_val['Predicted'], type='scatter', markersize=50, marker='^', color='red', panel=0)
                    ]
                    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
                    s  = mpf.make_mpf_style(marketcolors=mc)
                    # Plot volume if available; otherwise, volume=False
                    volume_flag = True if 'Volume' in df_val.columns else False
                    fig_val, axlist_val = mpf.plot(df_val, type='candle', style=s, volume=volume_flag,
                                                   addplot=apds_val, returnfig=True, figsize=(14,10), panel_ratios=(3,1))
                    fig_val.suptitle(f"Validation Set - {model_name}: Candlestick Chart with Predictions & RSI", fontsize=16)
                    val_candle_filename = f"{csv_base}_{forecast_history}_{forecast_horizon}_val_candlestick.png"
                    val_candle_path = result_history_dir / val_candle_filename
                    fig_val.savefig(val_candle_path)
                    plt.close(fig_val)
                    log_print(f"Validation candlestick chart with predictions saved to '{val_candle_path}'")
                except Exception as e:
                    log_print(f"Error during validation candlestick plotting: {e}")

            else:
                # For RSI target, a simpler plot is produced.
                plt.figure(figsize=(14,6))
                plt.plot(indices_val, actual_val, label='Actual RSI', marker='o', linestyle='-', color='blue')
                plt.plot(indices_val, preds_val, label='Predicted RSI', marker='x', linestyle='--', color='red')
                plt.xlabel('Sample Index', fontsize=14)
                plt.ylabel('RSI', fontsize=14)
                plt.title(f"Validation Set - {model_name}: RSI Comparison", fontsize=16)
                plt.legend(fontsize=12)
                plt.grid(True)
                plt.tight_layout()
                val_plot_filename = f"{csv_base}_{forecast_history}_{forecast_horizon}_val_rsi_plot.png"
                val_plot_path = result_history_dir / val_plot_filename
                plt.savefig(val_plot_path)
                plt.close()
                log_print(f"Validation RSI plot saved to '{val_plot_path}'")
        except Exception as e:
            log_print(f"Error during validation evaluation: {e}")

        log_print(f"\nEvaluation completed for '{csv_base}' with model '{model_name}' (Target: {target}) and forecast_history={forecast_history}.")
        log_print(f"Metrics and plots appended to '{metrics_path}' and saved in '{result_history_dir}'")
