import os
from pathlib import Path
import pandas as pd
import sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, \
    accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import mplfinance as mpf


def compute_rsi(prices, period=14):
    prices = np.array(prices)
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.empty_like(prices)
    rsi[:period] = np.nan
    rsi[period - 1] = 100. - 100. / (1. + rs)
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
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
    prices = pd.Series(np.array(prices).flatten())
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd.values, signal_line.values, histogram.values


def create_sequences(data, seq_length, horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length + horizon - 1])
    return np.array(X), np.array(y)


def evaluate_naive(csv_file_path,
                   model_name='Naive',
                   forecast_history=24,
                   forecast_horizon=1,
                   batch_size=64,  # not used
                   valid_size=0.05,  # not used
                   test_size=0.05,
                   include_baseline=True,
                   model_history_dir=None,
                   result_history_dir=None):
    """
    Evaluate the Naive Forecast model on the given CSV file.
    The naive forecast model predicts the last observed value as the forecast.
    Computes regression metrics, directional metrics, and generates plots:
      - Actual vs. Predicted Prices,
      - RSI Comparison, and
      - Candlestick Chart with Predictions & RSI.
    """
    csv_base = Path(csv_file_path).stem
    metrics_path = Path(result_history_dir) / f"{csv_base}_metrics.txt"

    # Load and preprocess CSV
    df_raw = pd.read_csv(csv_file_path)
    datetime_col = "Date"
    df_raw[datetime_col] = pd.to_datetime(df_raw[datetime_col])
    df = df_raw.sort_values(by=datetime_col)

    # Split data: Use the last 5% as the test set.
    total_size = len(df)
    test_split = int(total_size * test_size)
    train_df = df.iloc[:total_size - test_split]
    test_df = df.iloc[total_size - test_split:]

    # Fit scaler on training data
    scale_columns = ['Price']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[scale_columns])
    test_scaled = scaler.transform(test_df[scale_columns])

    # Create sequences for the test set
    X_test, y_test = create_sequences(test_scaled, forecast_history, forecast_horizon)
    # Naive forecast: prediction = last value of each sequence
    naive_preds = X_test[:, -1, 0]
    # Inverse transform
    naive_preds_inv = scaler.inverse_transform(naive_preds.reshape(-1, 1)).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse_val = mean_squared_error(y_test_inv, naive_preds_inv)
    mae_val = mean_absolute_error(y_test_inv, naive_preds_inv)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(y_test_inv, naive_preds_inv)
    mape_val = mean_absolute_percentage_error(y_test_inv, naive_preds_inv)

    with open(metrics_path, 'a') as metrics_file:
        def log_print(*args, **kwargs):
            msg = ' '.join(map(str, args))
            print(msg)
            metrics_file.write(msg + '\n')

        log_print(f"\n--- Naive Evaluation at {pd.Timestamp.now()} ---")
        log_print("Naive Forecast Model")
        log_print(f"CSV Name: {csv_base}")
        log_print(f"Forecast History: {forecast_history}")
        log_print(f"Forecast Horizon: {forecast_horizon}")
        log_print("\nTest Regression Metrics:")
        log_print(f"  MAE  : {mae_val:.6f}")
        log_print(f"  MSE  : {mse_val:.6f}")
        log_print(f"  RMSE : {rmse_val:.6f}")
        log_print(f"  R2   : {r2_val:.6f}")
        log_print(f"  MAPE : {mape_val:.6f}")

        # Additional RSI and MACD Metrics
        if len(y_test_inv) >= 14:
            rsi_actual = compute_rsi(y_test_inv, period=14)
            rsi_pred = compute_rsi(naive_preds_inv, period=14)
            valid_idx = ~np.isnan(rsi_actual)
            if valid_idx.sum() > 0:
                rsi_mae = mean_absolute_error(rsi_actual[valid_idx], rsi_pred[valid_idx])
                rsi_mse = mean_squared_error(rsi_actual[valid_idx], rsi_pred[valid_idx])
                log_print("\nRSI Metrics:")
                log_print(f"  RSI MAE: {rsi_mae:.4f}")
                log_print(f"  RSI MSE: {rsi_mse:.4f}")
        if len(y_test_inv) >= 26:
            def compute_macd_local(prices, fast=12, slow=26, signal=9):
                prices = pd.Series(np.array(prices).flatten())
                ema_fast = prices.ewm(span=fast, adjust=False).mean()
                ema_slow = prices.ewm(span=slow, adjust=False).mean()
                macd = ema_fast - ema_slow
                signal_line = macd.ewm(span=signal, adjust=False).mean()
                histogram = macd - signal_line
                return macd.values, signal_line.values, histogram.values

            macd_actual, _, _ = compute_macd_local(y_test_inv)
            macd_pred, _, _ = compute_macd_local(naive_preds_inv)
            macd_mae = mean_absolute_error(macd_actual, macd_pred)
            log_print("\nMACD Metric:")
            log_print(f"  MACD MAE: {macd_mae:.4f}")

        # Direction Classification Metrics
        actual_dirs = []
        predicted_dirs = []
        for i in range(1, len(y_test_inv)):
            actual_dir = 1 if y_test_inv[i] > y_test_inv[i - 1] else 0
            pred_dir = 1 if naive_preds_inv[i] > y_test_inv[i - 1] else 0
            actual_dirs.append(actual_dir)
            predicted_dirs.append(pred_dir)
        acc_val = accuracy_score(actual_dirs, predicted_dirs)
        prec_val = precision_score(actual_dirs, predicted_dirs, zero_division=0)
        rec_val = recall_score(actual_dirs, predicted_dirs, zero_division=0)
        f1_val = f1_score(actual_dirs, predicted_dirs, zero_division=0)
        log_print("\nTest Direction Classification Metrics:")
        log_print(f"  Accuracy : {acc_val:.4f}")
        log_print(f"  Precision: {prec_val:.4f}")
        log_print(f"  Recall   : {rec_val:.4f}")
        log_print(f"  F1       : {f1_val:.4f}")

        # Plot: Actual vs. Predicted Prices (Test Set)
        try:
            indices = np.arange(len(y_test_inv))
            plt.figure(figsize=(14, 7))
            plt.plot(indices, y_test_inv, label='Actual Price', marker='o', linestyle='-', color='blue')
            plt.plot(indices, naive_preds_inv, label="Naive Predictions", marker='x', linestyle='--', color='red')
            plt.xlabel('Sample Index', fontsize=14)
            plt.ylabel('Price', fontsize=14)
            plt.title("Test Set - Naive: Actual vs. Predicted Prices", fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            price_plot_filename = f"{csv_base}_{forecast_history}_{forecast_horizon}_test_price_plot_naive.png"
            price_plot_path = Path(result_history_dir) / price_plot_filename
            plt.savefig(price_plot_path)
            plt.close()
            log_print(f"Test price plot saved to '{price_plot_path}'")
        except Exception as e:
            log_print(f"Error during test price plotting: {e}")

        # Plot: RSI Comparison (Test Set)
        try:
            indices = np.arange(len(y_test_inv))
            plt.figure(figsize=(14, 7))
            rsi_actual = compute_rsi(y_test_inv, period=14)
            rsi_pred = compute_rsi(naive_preds_inv, period=14)
            plt.plot(indices, rsi_actual, label='Actual RSI', marker='o', linestyle='-', color='blue')
            plt.plot(indices, rsi_pred, label="Naive Predicted RSI", marker='x', linestyle='--', color='red')
            plt.xlabel('Sample Index', fontsize=14)
            plt.ylabel('RSI', fontsize=14)
            plt.title("Test Set - Naive: RSI Comparison", fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            rsi_plot_filename = f"{csv_base}_{forecast_history}_{forecast_horizon}_test_rsi_plot_naive.png"
            rsi_plot_path = Path(result_history_dir) / rsi_plot_filename
            plt.savefig(rsi_plot_path)
            plt.close()
            log_print(f"Test RSI comparison plot saved to '{rsi_plot_path}'")
        except Exception as e:
            log_print(f"Error during test RSI plotting: {e}")

        # Test Candlestick Chart with Predictions & RSI
        try:
            df_candle = df_raw.copy()
            df_candle[datetime_col] = pd.to_datetime(df_candle[datetime_col])
            df_candle = df_candle.sort_values(by=datetime_col)
            if 'Price' in df_candle.columns:
                df_candle.rename(columns={'Price': 'Close'}, inplace=True)
            if 'Vol.' in df_candle.columns:
                df_candle.rename(columns={'Vol.': 'Volume'}, inplace=True)
                df_candle['Volume'] = df_candle['Volume'].apply(
                    lambda x: float(x.replace('K', '')) * 1000 if isinstance(x, str) and 'K' in x else float(x)
                )
            df_candle.set_index(datetime_col, inplace=True)
            rsi_values = compute_rsi(df_candle['Close'].values, period=14)
            df_candle['RSI'] = rsi_values
            test_index = pd.to_datetime(test_df[datetime_col])
            n_missing = forecast_history + forecast_horizon - 1
            pred_full = [np.nan] * n_missing + list(naive_preds_inv.flatten())
            pred_full = pred_full[:len(test_index)]
            pred_series = pd.Series(pred_full, index=test_index)
            df_candle.loc[test_index, 'Predicted'] = pred_series
            apds = [
                mpf.make_addplot(df_candle['RSI'], panel=1, color='blue', ylabel='RSI'),
                mpf.make_addplot(df_candle['Predicted'], type='scatter', markersize=50, marker='^', color='red',
                                 panel=0)
            ]
            mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
            style = mpf.make_mpf_style(marketcolors=mc)
            volume_flag = True if 'Volume' in df_candle.columns else False
            fig, axlist = mpf.plot(df_candle, type='candle', style=style, volume=volume_flag,
                                   addplot=apds, returnfig=True, figsize=(14, 10), panel_ratios=(3, 1))
            fig.suptitle("Test Set - Naive: Candlestick Chart with Predictions & RSI", fontsize=16)
            candle_plot_filename = f"{csv_base}_{forecast_history}_{forecast_horizon}_candlestick_naive.png"
            candle_plot_path = Path(result_history_dir) / candle_plot_filename
            fig.savefig(candle_plot_path)
            plt.close(fig)
            log_print(f"Test candlestick chart saved to '{candle_plot_path}'")
        except Exception as e:
            log_print(f"Error during test candlestick plotting: {e}")

        log_print(
            f"\nEvaluation completed for '{csv_base}' with model 'Naive' and forecast_history={forecast_history}.")
        log_print(f"Metrics and plots appended to '{metrics_path}' and saved in '{result_history_dir}'")
