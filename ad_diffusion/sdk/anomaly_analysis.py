from __future__ import annotations

import logging
import os
import sys
from pathlib import Path  # noqa: TC003

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sdk.inference_ad import (
    _resolve_model_paths,
    get_model_target_dim,
    inference_ad_tesseract2_mp,
)
from sdk.thresholds import MACSThresholdStrategy, SCSThresholdStrategy

# Set up logging
logger = logging.getLogger(__name__)


def perform_anomaly_analysis_with_diffusion(
    df: pd.DataFrame,
    *,
    threshold_strategy: str,
    model_path: str | Path | None = None,
    config_path: str | Path = "",
    nsample: int = 15,
    preprocess_model_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Perform anomaly analysis using Tesseract AD Diffusion Model.

    If ``model_path``/``config_path`` do not exist locally, the default weights
    (``final_model.pth`` + ``curriculum_medium.yaml``) are automatically
    downloaded from the Hugging Face repository
    ``nvidia/nv-tesseract-ad-diffusion``.

    Args:
        df: DataFrame containing numeric data
        threshold_strategy: Strategy to use for threshold calculation ('scs' or 'macs')
        model_path: Path to the diffusion model checkpoint. If ``None`` or missing
            locally, the default checkpoint is downloaded from Hugging Face.
        config_path: Path to the model config file (optional if config is in checkpoint)
        nsample: Number of samples for diffusion model inference
        preprocess_model_dir: Directory containing preprocessing model (optional)

    Returns:
        DataFrame with original data and anomaly detection results
    """
    # Prepare data for diffusion model
    # The diffusion model expects all numeric columns
    input_df = df.copy()

    # Validate all columns are numeric
    non_numeric_cols = []
    for col in input_df.columns:
        try:
            input_df[col] = pd.to_numeric(input_df[col], errors="raise")
        except (ValueError, TypeError):
            non_numeric_cols.append(col)

    if non_numeric_cols:
        raise ValueError(
            f"The following columns contain non-numeric values: {non_numeric_cols}. "
            f"All input values must be numeric for anomaly detection."
        )

    # Resolve / auto-download weights once up front so downstream calls share them.
    resolved_model, resolved_config = _resolve_model_paths(
        str(model_path) if model_path else None,
        str(config_path) if config_path else "",
    )

    # Get target_dim from model and validate data size BEFORE running inference
    target_dim = get_model_target_dim(resolved_model, resolved_config)
    n_samples = len(input_df)

    if n_samples < target_dim:
        raise ValueError(
            f"Insufficient data samples for PCA: got {n_samples} samples but model requires "
            f"at least {target_dim} samples (target_dim={target_dim}). "
            f"Please provide more data."
        )

    # Run inference with diffusion model (auto-uses multi-GPU when available)
    results = inference_ad_tesseract2_mp(
        data=input_df,
        model_path=resolved_model,
        config_path=resolved_config,
        nsample=nsample,
        preprocess_model_dir=str(preprocess_model_dir) if preprocess_model_dir else None,
    )

    # Extract residual scores (MAE) from results
    residual_scores = results["residual"]

    # Get target data for advanced thresholding methods
    target_data = results["target"]

    # Apply thresholding strategy
    if threshold_strategy == "scs":
        # Use actual target data from the model results
        anomalies = SCSThresholdStrategy().scs_thresholder.detect_anomalies(residual_scores, target_data)
    elif threshold_strategy == "macs":
        # Use actual target data from the model results
        anomalies = MACSThresholdStrategy().macs_thresholder.detect_anomalies(residual_scores, target_data)
    else:
        raise ValueError(f"Unknown threshold strategy: {threshold_strategy}")

    # Create result dataframe
    result_df = df.copy()

    # Ensure consistent lengths - model outputs may be padded
    original_length = len(df)
    if len(residual_scores) != original_length:
        logger.info(f"Aligning lengths: residual_scores={len(residual_scores)}, original_data={original_length}")
        # Truncate all arrays to match original dataframe length
        residual_scores = residual_scores[:original_length]
        anomalies = anomalies[:original_length]
        # Also truncate target_data if it's used later
        if len(target_data) > original_length:
            target_data = target_data[:original_length]

    result_df["Anomaly"] = anomalies
    result_df["MAE"] = residual_scores  # Using residual (MAE) as anomaly score

    return result_df
