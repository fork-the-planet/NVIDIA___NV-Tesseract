#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quick example showing how to load a CSV file and perform forecasting.

This example demonstrates:
1. Auto-downloading model weights from Hugging Face (on first run)
2. Standard forecasting mode
3. DARR mode (context-enhanced forecasting)
4. Interpretability mode (lag x horizon attributions, JSON + PDF report;
   multivariate panels use channel-flow GPU batching and optional multi-GPU
   Shapley coupling via SDK parallelism flags)

Model weights download automatically from the public Hugging Face repo on first run.

Multi-GPU interpretability (2+ CUDA devices):
    RUN_MULTI_GPU_INTERP=1 uv run python quick_example.py
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from sdk.forecasting import perform_forecasting

logger = logging.getLogger(__name__)

# Load your CSV file
csv_path = (
    Path(__file__).resolve().parent / "tests" / "datasets" / "ETTh_single_feature.csv"
)  # Replace with your file path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not os.path.exists(csv_path):
        logger.error("CSV file not found at %s", csv_path)
        logger.error("Please supply your own data with 'timestamp' and target columns.")
        logger.error("Example CSV format:")
        logger.error("timestamp,target")
        logger.error("2024-01-01 00:00:00,1.5")
        logger.error("2024-01-01 01:00:00,1.7")
        logger.error("...")
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
    logger.info(
        "\nStandard forecast (only predicted rows with '%s_forecast' column):\n%s", target_col, forecast_df.to_csv()
    )

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
    logger.info("\nDARR forecast (hybrid prediction in '%s_forecast' column):\n%s", target_col, darr_df.to_csv())

    # Interpretability mode: produces a lag x horizon attribution heatmap and a
    # multi-page PDF report alongside the forecast. Set ``interpretability=True``
    # and choose the output format via ``interpretability_output``:
    #   - "json" -> writes only forecast.csv + explanation.json
    #   - "pdf"  -> writes forecast.csv + lag_horizon_*.csv/png + explanation_report.pdf
    #   - None   -> writes both (full bundle)
    interp_out_dir = Path(__file__).resolve().parent / "interpretability_output"
    # Use a multivariate panel so channel-flow batching and coupling run (C > 1).
    interp_csv_path = Path(__file__).resolve().parent / "tests" / "datasets" / "ETTh_4feature.csv"
    if not interp_csv_path.exists():
        raise SystemExit(f"Multivariate CSV not found at {interp_csv_path} (needed for channel-flow demo).")
    df_interp = pd.read_csv(interp_csv_path)

    multi_gpu = os.environ.get("RUN_MULTI_GPU_INTERP", "").strip() in ("1", "true", "yes")
    shapley_workers = 2 if multi_gpu else 0
    parallel_passes = multi_gpu

    interp_df = perform_forecasting(
        df=df_interp,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        timestamp_column=timestamp_col,
        target_column=target_col,
        # Interpretability controls
        interpretability=True,
        interpretability_output=None,  # write both JSON and PDF
        interpretability_out_dir=interp_out_dir,
        interpretability_dataset_name=interp_csv_path.name,
        n_lags=128,
        softmax_tau=1.0,
        # Pass A: Jacobian channel flow (GPU-batched across transitions)
        interpretability_channel_axis=True,
        interpretability_transition_batch=8,
        interpretability_channel_batch_size=64,
        # Pass B: Shapley coupling matrix (auto-enabled for C > 1)
        interpretability_coupling=True,
        interpretability_coupling_transitions=16,
        interpretability_shapley_n_samples=64,
        interpretability_shapley_baseline="zero",
        # Multi-GPU: shard Pass B and/or overlap Pass A + Pass B (needs 2+ CUDA devices)
        interpretability_devices="auto",
        interpretability_parallel_passes=parallel_passes,
        interpretability_shapley_workers=shapley_workers,
        save_preds="forecast_ETTh_with_explanations.csv",
    )
    logger.info(
        "\nInterpretability forecast (single-window baseline in '%s_forecast' column):\n%s",
        target_col,
        interp_df.head().to_string(index=False),
    )
    logger.info("\nReport bundle written under: %s", interp_out_dir)
    logger.info("  (includes channel_coupling_matrix.csv when coupling is enabled)")

    if multi_gpu:
        logger.info("Multi-GPU interpretability ran via SDK parallelism flags.")
    else:
        oss_root = Path(__file__).resolve().parent.parent
        logger.info(
            "\nOptional: benchmark multi-GPU Shapley sharding (standalone CLI):\n"
            "  cd %s\n"
            "  uv run python run_interpretability_sharded.py --benchmark --shapley-workers 2\n"
            "Or re-run this script with RUN_MULTI_GPU_INTERP=1 on a 2+ GPU host.",
            oss_root,
        )
        if os.environ.get("RUN_SHARDED_CLI"):
            subprocess.run(
                [
                    sys.executable,
                    str(oss_root / "run_interpretability_sharded.py"),
                    "--csv",
                    str(interp_csv_path),
                    "--seq-len",
                    str(seq_len),
                    "--forecast-horizon",
                    str(forecast_horizon),
                    "--shapley-workers",
                    "2",
                ],
                cwd=str(oss_root),
                check=False,
            )
