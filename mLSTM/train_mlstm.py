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

from xLSTM.xLSTM import mLSTM  # Import the mLSTM class from xLSTM.py


def train_mlstm(csv_file_path, model_name='mLSTM', max_iterations=1,
                forecast_history=24, forecast_horizon=1, batch_size=64,
                valid_size=0.05, test_size=0.05, include_baseline=True,
                model_history_dir=None, result_history_dir=None):
    """
    Train the mLSTM model on the given CSV file with specified parameters.
    The training procedure includes splitting the data into train, validation, and test sets,
    scaling the data, creating sequences, and running multiple training iterations.
    The best model (lowest validation MSE) is saved.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure CSV file exists
    if not os.path.isfile(csv_file_path):
        print(f"Error: The file '{csv_file_path}' does not exist.")
        return
    csv_base = Path(csv_file_path).stem
    if model_history_dir is None or result_history_dir is None:
        print("Error: 'model_history_dir' and 'result_history_dir' must be provided.")
        return

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
            ('drop_duplicates',
             sklearn.preprocessing.FunctionTransformer(lambda df: df.drop_duplicates(subset=datetime_col))),
            ('fill_missing',
             sklearn.preprocessing.FunctionTransformer(lambda df: df.fillna(method=method).fillna(value=value)))
        ])
        df = preproc_pipe.fit_transform(df_raw)
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(by=datetime_col)
        log_print("Data preprocessing completed.")

        # Split data
        total_size = len(df)
        test_split = int(total_size * test_size)
        valid_split = int(total_size * valid_size)
        train_end = total_size - test_split - valid_split
        valid_end = total_size - test_split
        train_df = df.iloc[:train_end]
        valid_df = df.iloc[train_end:valid_end]
        test_df = df.iloc[valid_end:]
        log_print(
            f"Data split into train ({len(train_df)}), validation ({len(valid_df)}), and test ({len(test_df)}) sets.")

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

        # Define mLSTM model
        input_size = 1
        head_size = 32
        num_heads = 2
        num_layers = 3
        model = mLSTM(input_size, head_size, num_heads, num_layers=num_layers, batch_first=True).to(device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for iteration in range(max_iterations):
            log_print(f"\nStarting training iteration {iteration + 1}/{max_iterations}")
            model.train()
            epoch_loss = 0
            for inputs, targets in tqdm(train_loader, desc=f"Training Iteration {iteration + 1}"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs, _ = model(inputs)
                outputs = outputs[:, -1, :]
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / len(train_loader)
            log_print(f"Iteration {iteration + 1}: Average Training Loss: {avg_epoch_loss:.6f}")

            # Validate model
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs, _ = model(inputs)
                    outputs = outputs[:, -1, :]
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(valid_loader)
            log_print(f"Iteration {iteration + 1}: Average Validation Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_mse:
                best_mse = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                log_print(f"New best model saved at '{best_model_path}' with Validation MSE {best_mse:.6f}")

        # Evaluate on test set
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        test_predictions = []
        actuals = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs, _ = model(inputs)
                outputs = outputs[:, -1, :]
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
