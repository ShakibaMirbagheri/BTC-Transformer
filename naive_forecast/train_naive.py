import os
from pathlib import Path
import torch


def train_naive(csv_file_path, model_name='Naive', max_iterations=1,
                forecast_history=24, forecast_horizon=1, batch_size=64,
                valid_size=0.05, test_size=0.05, include_baseline=True,
                model_history_dir=None, result_history_dir=None):
    """
    "Train" the Naive Forecast model.
    The naive forecast model does not require training.
    This function simply writes a dummy model file.
    """
    if model_history_dir is None or result_history_dir is None:
        print("Error: 'model_history_dir' and 'result_history_dir' must be provided.")
        return
    csv_base = Path(csv_file_path).stem
    best_model_filename = f"{csv_base}_best_model.pt"
    best_model_path = model_history_dir / best_model_filename
    # Save a dummy dictionary to indicate that this is a naive model.
    torch.save({'naive': True}, best_model_path)
    print(f"Naive forecast model for '{csv_base}' saved at '{best_model_path}'")
