#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Quick example showing how to load a CSV file and perform forecasting.

This example demonstrates:
1. Auto-downloading model weights from Hugging Face (on first run)
2. Standard forecasting mode
3. DARR mode (context-enhanced forecasting)

Make sure you're authenticated with Hugging Face for the private repo:
    huggingface-cli login
"""

import os
from pathlib import Path

import pandas as pd

from sdk.forecasting import perform_forecasting

# Load your CSV file
csv_path = (
    Path(__file__).resolve().parent / "tests" / "datasets" / "ETTh_single_feature.csv"
)  # Replace with your file path

if __name__ == "__main__":
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        print("Please supply your own data with 'timestamp' and target columns.")
        print("Example CSV format:")
        print("timestamp,target")
        print("2024-01-01 00:00:00,1.5")
        print("2024-01-01 01:00:00,1.7")
        print("...")
        raise SystemExit("Data file required")

    # Model weights will be auto-downloaded from Hugging Face on first run
    # You can also specify custom paths if you have the files locally
    df = pd.read_csv(csv_path)
    timestamp_col = "timestamp"
    target_col = "LULL"
    seq_len = 512
    forecast_horizon = 100

    # Standard forecasting (no external memory)
    # Model weights will be auto-downloaded if not present
    forecast_df = perform_forecasting(
        df=df,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        timestamp_column=timestamp_col,
        target_column=target_col,
        save_preds="forecast_ETTh_seq_len_100.csv",
    )
    print(f"\nStandard forecast (only predicted rows with '{target_col}_forecast' column):")
    print(forecast_df.to_csv())

    # DARR mode requires a user-provided context dataset that mirrors the input schema
    # but contains different data (e.g., historical slices, previously predicted values, etc.).
    # This example loads one such CSV to demonstrate how to run DARR.
    context_csv_path = Path(__file__).resolve().parent / "tests" / "datasets" / "ETTh_single_feature_darr_context.csv"
    if not context_csv_path.exists():
        raise SystemExit(f"Context CSV not found at {context_csv_path}, please provide your own for DARR mode.")

    context_df = pd.read_csv(context_csv_path)
    darr_df = perform_forecasting(
        df=df,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        context_df=context_df,  # This enables DARR mode
        timestamp_column=timestamp_col,
        target_column=target_col,
        save_preds="forecast_ETTh_darr_100.csv",
        # Model weights auto-downloaded if needed
    )
    print(f"\nDARR forecast (hybrid prediction in '{target_col}_forecast' column):")
    print(darr_df.to_csv())
