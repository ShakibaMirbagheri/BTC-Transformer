import os
from pathlib import Path
import pandas as pd
import sklearn
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import mplfinance as mpf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_percentage_error
)


def compute_rsi(prices, period=14):
    """
    Compute the Relative Strength Index (RSI) for a series of prices.
    Returns a 1D numpy array of RSI values. The first (period-1) values will be NaN.
    """
    prices = np.array(prices)
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
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
    """
    Compute the MACD (Moving Average Convergence Divergence) for a series of prices.
    Returns a tuple of three 1D numpy arrays: (macd, signal_line, histogram).
    """
    prices = pd.Series(np.array(prices).flatten())
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd.values, signal_line.values, histogram.values


def train_lstm(csv_file_path, model_name='LSTM', max_iterations=1,
              forecast_history=24, forecast_horizon=1, batch_size=64,
              valid_size=0.05, test_size=0.05, include_baseline=True,
              model_history_dir=None, result_history_dir=None):
    """
    Train the standard LSTM model on the given CSV file with specified parameters.

    Parameters:
        csv_file_path (str): Path to the CSV file.
        model_name (str): Name of the model (e.g., 'LSTM').
        max_iterations (int): Number of training iterations.
        forecast_history (int): Number of past time steps to use.
        forecast_horizon (int): Number of future time steps to predict.
        batch_size (int): Batch size for training.
        valid_size (float): Fraction of data to use for validation.
        test_size (float): Fraction of data to use for testing.
        include_baseline (bool): Whether to include baseline in metrics and plots.
        model_history_dir (Path): Directory to save models for the current history.
        result_history_dir (Path): Directory to save results for the current history.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    best_model_filename = f"{csv_base}_best_model.pt"
    best_model_path = model_history_dir / best_model_filename
    metrics_path = result_history_dir / f"{csv_base}_metrics.txt"

    best_mse = float('inf')

    with open(metrics_path, 'w') as metrics_file:
        def log_print(*args, **kwargs):
            msg = ' '.join(map(str, args))
            print(msg)
            metrics_file.write(msg + '\n')

        log_print(f"Model Name: {model_name}")
        log_print(f"CSV Name: {csv_base}")
        log_print(f"Training Started at: {pd.Timestamp.now()}")
        log_print(f"Sequence Length (Forecast History): {forecast_history}")
        log_print(f"Forecast Horizon: {forecast_horizon}")

        # Load CSV
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

        preproc_pipe = sklearn.pipeline.Pipeline([
            ('drop_duplicates', sklearn.preprocessing.FunctionTransformer(lambda df: df.drop_duplicates(subset=datetime_col))),
            ('fill_missing', sklearn.preprocessing.FunctionTransformer(lambda df: df.fillna(method=method).fillna(value=value)))
        ])

        df = preproc_pipe.fit_transform(df_raw)
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(by=datetime_col)
        log_print("Data preprocessing completed.")

        # Split data into train, validation, and test sets
        total_size = len(df)
        test_split = int(total_size * test_size)
        valid_split = int(total_size * valid_size)
        train_end = total_size - test_split - valid_split
        valid_end = total_size - test_split
        train_df = df.iloc[:train_end]
        valid_df = df.iloc[train_end:valid_end]
        test_df = df.iloc[valid_end:]
        log_print(f"Data split into train ({len(train_df)}), validation ({len(valid_df)}), and test ({len(test_df)}) sets.")

        scale_columns = ['Price']
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_df[scale_columns])
        train_scaled = scaler.transform(train_df[scale_columns])
        valid_scaled = scaler.transform(valid_df[scale_columns])
        test_scaled = scaler.transform(test_df[scale_columns])
        log_print("Data scaling completed.")

        # Create sequences
        def create_sequences(data, seq_length, horizon):
            X, y = [], []
            for i in range(len(data) - seq_length - horizon + 1):
                X.append(data[i:(i + seq_length)])
                y.append(data[i + seq_length + horizon - 1])
            return torch.Tensor(X), torch.Tensor(y)

        trainX, trainY = create_sequences(train_scaled, forecast_history, forecast_horizon)
        validX, validY = create_sequences(valid_scaled, forecast_history, forecast_horizon)
        testX, testY = create_sequences(test_scaled, forecast_history, forecast_horizon)
        log_print("Prepared forecasting data.")

        train_dataset = TensorDataset(trainX, trainY)
        valid_dataset = TensorDataset(validX, validY)
        test_dataset = TensorDataset(testX, testY)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Define the standard LSTM model
        input_size = 1
        hidden_size = 64
        num_layers = 3
        model = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True).to(device)

        # Define a fully connected layer to map LSTM outputs to the desired output size
        fc = nn.Linear(hidden_size, 1).to(device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(fc.parameters()), lr=0.001)

        for iteration in range(max_iterations):
            log_print(f"\nStarting training iteration {iteration + 1}/{max_iterations}")

            model.train()
            fc.train()
            epoch_loss = 0
            for inputs, targets in tqdm(train_loader, desc=f"Training Iteration {iteration + 1}"):
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs, _ = model(inputs)
                outputs = outputs[:, -1, :]  # Take the last output
                outputs = fc(outputs)  # Map to desired output size
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            log_print(f"Iteration {iteration + 1}: Average Training Loss: {avg_epoch_loss:.6f}")

            # Validate the model
            model.eval()
            fc.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs, _ = model(inputs)
                    outputs = outputs[:, -1, :]
                    outputs = fc(outputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(valid_loader)
            log_print(f"Iteration {iteration + 1}: Average Validation Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_mse:
                best_mse = avg_val_loss
                torch.save({
                    'lstm_state_dict': model.state_dict(),
                    'fc_state_dict': fc.state_dict()
                }, best_model_path)
                log_print(f"New best model saved at '{best_model_path}' with Validation MSE {best_mse:.6f}")

        # After training, evaluate on the test set
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['lstm_state_dict'])
        fc.load_state_dict(checkpoint['fc_state_dict'])
        model.eval()
        fc.eval()
        test_predictions = []
        actuals = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs, _ = model(inputs)
                outputs = outputs[:, -1, :]
                outputs = fc(outputs)
                test_predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())

        test_predictions = scaler.inverse_transform(np.array(test_predictions))
        actuals = scaler.inverse_transform(np.array(actuals))

        mse_val = mean_squared_error(actuals, test_predictions)
        mae_val = mean_absolute_error(actuals, test_predictions)
        rmse_val = np.sqrt(mse_val)

        log_print(f"\nFinal Test MSE: {mse_val:.6f}")
        log_print(f"Final Test MAE: {mae_val:.6f}")

        if include_baseline:
            baseline_predictions = []
            for i in range(len(testX)):
                baseline_predictions.append(testX[i][-1][-1].item())
            baseline_predictions = scaler.inverse_transform(np.array(baseline_predictions).reshape(-1, 1))
            baseline_mse = mean_squared_error(actuals, baseline_predictions)
            baseline_mae = mean_absolute_error(actuals, baseline_predictions)
            baseline_rmse = np.sqrt(baseline_mse)
            log_print("\nBaseline Metrics:")
            log_print(f"Baseline Test MSE: {baseline_mse:.6f}")
            log_print(f"Baseline Test MAE: {baseline_mae:.6f}")
            mse_diff = mse_val - baseline_mse
            mse_percentage = (mse_diff / baseline_mse) * 100
            if mse_percentage > 0:
                log_print(f"Model MSE is worse by {mse_percentage:.2f}% compared to the baseline.")
            else:
                log_print(f"Model MSE is better by {abs(mse_percentage):.2f}% compared to the baseline.")

        log_print(f"\nTest Metrics saved at '{metrics_path}'")

        # --- Additional RSI and MACD Metrics ---
        if len(actuals) >= 14:
            rsi_actual = compute_rsi(actuals, period=14)
            rsi_pred = compute_rsi(test_predictions, period=14)
            valid_idx = ~np.isnan(rsi_actual)
            if valid_idx.sum() > 0:
                rsi_mae = mean_absolute_error(rsi_actual[valid_idx], rsi_pred[valid_idx])
                rsi_mse = mean_squared_error(rsi_actual[valid_idx], rsi_pred[valid_idx])
                log_print("\nRSI Metrics:")
                log_print(f"  RSI MAE: {rsi_mae:.4f}")
                log_print(f"  RSI MSE: {rsi_mse:.4f}")
        if len(actuals) >= 26:
            def compute_macd_local(prices, fast=12, slow=26, signal=9):
                prices = pd.Series(np.array(prices).flatten())
                ema_fast = prices.ewm(span=fast, adjust=False).mean()
                ema_slow = prices.ewm(span=slow, adjust=False).mean()
                macd = ema_fast - ema_slow
                signal_line = macd.ewm(span=signal, adjust=False).mean()
                histogram = macd - signal_line
                return macd.values, signal_line.values, histogram.values
            macd_actual, _, _ = compute_macd_local(actuals)
            macd_pred, _, _ = compute_macd_local(test_predictions)
            macd_mae = mean_absolute_error(macd_actual, macd_pred)
            log_print("\nMACD Metric:")
            log_print(f"  MACD MAE: {macd_mae:.4f}")
        # ----------------------------------------

        # Direction Classification Metrics (Test)
        actual_dirs = []
        predicted_dirs = []
        for i in range(1, len(actuals)):
            actual_dir = 1 if actuals[i] > actuals[i - 1] else 0
            pred_dir = 1 if test_predictions[i] > actuals[i - 1] else 0
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
            indices = np.arange(len(actuals))
            plt.figure(figsize=(14, 7))
            plt.plot(indices, actuals, label='Actual Price', marker='o', linestyle='-', color='blue')
            plt.plot(indices, test_predictions, label=f"{model_name} Predictions", marker='x', linestyle='--', color='red')
            if include_baseline:
                plt.plot(indices, baseline_predictions, label='Baseline Prediction', marker='s', linestyle=':', color='green')
            plt.xlabel('Sample Index', fontsize=14)
            plt.ylabel('Price', fontsize=14)
            plt.title(f"Test Set - {model_name}: Actual vs. Predicted Prices", fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            price_plot_filename = f"{csv_base}_{forecast_history}_{forecast_horizon}_test_price_plot.png"
            price_plot_path = result_history_dir / price_plot_filename
            plt.savefig(price_plot_path)
            plt.close()
            log_print(f"Test price plot saved to '{price_plot_path}'")
        except Exception as e:
            log_print(f"Error during test price plotting: {e}")

        # Plot: RSI Comparison (Test Set)
        try:
            indices = np.arange(len(actuals))
            plt.figure(figsize=(14, 7))
            rsi_actual = compute_rsi(actuals, period=14)
            rsi_pred = compute_rsi(test_predictions, period=14)
            plt.plot(indices, rsi_actual, label='Actual RSI', marker='o', linestyle='-', color='blue')
            plt.plot(indices, rsi_pred, label=f"{model_name} Predicted RSI", marker='x', linestyle='--', color='red')
            plt.xlabel('Sample Index', fontsize=14)
            plt.ylabel('RSI', fontsize=14)
            plt.title(f"Test Set - {model_name}: RSI Comparison", fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            rsi_plot_filename = f"{csv_base}_{forecast_history}_{forecast_horizon}_test_rsi_plot.png"
            rsi_plot_path = result_history_dir / rsi_plot_filename
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
            # Use the Date column from test_df as the index for predictions
            test_index = pd.to_datetime(test_df[datetime_col])
            n_missing = forecast_history + forecast_horizon - 1
            pred_full = [np.nan] * n_missing + list(test_predictions.flatten())
            pred_full = pred_full[:len(test_index)]
            pred_series = pd.Series(pred_full, index=test_index)
            df_candle.loc[test_index, 'Predicted'] = pred_series
            apds = [
                mpf.make_addplot(df_candle['RSI'], panel=1, color='blue', ylabel='RSI'),
                mpf.make_addplot(df_candle['Predicted'], type='scatter', markersize=50, marker='^', color='red', panel=0)
            ]
            mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
            style = mpf.make_mpf_style(marketcolors=mc)
            volume_flag = True if 'Volume' in df_candle.columns else False
            fig, axlist = mpf.plot(df_candle, type='candle', style=style, volume=volume_flag,
                                   addplot=apds, returnfig=True, figsize=(14, 10), panel_ratios=(3, 1))
            fig.suptitle(f"Test Set - {model_name}: Candlestick Chart with Predictions & RSI", fontsize=16)
            candle_plot_filename = f"{csv_base}_{forecast_history}_{forecast_horizon}_candlestick.png"
            candle_plot_path = result_history_dir / candle_plot_filename
            fig.savefig(candle_plot_path)
            plt.close(fig)
            log_print(f"Test candlestick chart saved to '{candle_plot_path}'")
        except Exception as e:
            log_print(f"Error during test candlestick plotting: {e}")

        log_print(f"\nEvaluation completed for '{csv_base}' with model '{model_name}' and forecast_history={forecast_history}.")
        log_print(f"Metrics and plots appended to '{metrics_path}' and saved in '{result_history_dir}'")
