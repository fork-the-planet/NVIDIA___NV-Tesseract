"""Single inference HPO trial for TAO AutoML integration.

Called once per AutoML recommendation by AutoMLRunner via VirtualEnvSDK:
    python inference_hpo_trial.py --run-config {config_path}

Reads hyperparameters from a YAML config, runs diffusion inference on a
labeled CSV, and writes {"f1_score": <float>} to metrics.json so the
AutoML runner can extract the metric.

Config keys consumed:
    dataset.csv            - labeled CSV (must include a ground-truth label column)
    dataset.label_col      - name of the ground-truth column (default: label)
    dataset.eval_rows      - rows from the top of csv to evaluate (default: 2000)
    inference.nsample      - diffusion samples per window (tunable by AutoML)
    inference.threshold_strategy - "scs" or "macs" (fixed per AutoML pass)
    inference.model_path   - path to checkpoint; empty = HF auto-download
    inference.config_path  - path to config YAML; empty = HF auto-download
    train.output_dir       - directory where metrics.json is written
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
import pandas as pd
from sklearn.metrics import f1_score

AD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AD_DIR))

import sdk.inference_ad as _inf  # patch device before importing analysis module
if torch.backends.mps.is_available():
    _inf.DEVICE = "mps"  # inference_ad defaults to cuda→cpu, skipping MPS

from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-config", required=True, help="YAML config written by AutoMLRunner")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.run_config).read_text())
    ds  = cfg.get("dataset", {})
    inf = cfg.get("inference", {})
    out = Path(cfg.get("train", {}).get("output_dir", "artifacts/inference_hpo"))
    out.mkdir(parents=True, exist_ok=True)

    csv_path  = ds["csv"]
    label_col = ds.get("label_col", "label")
    eval_rows = int(ds.get("eval_rows", 2000))
    nsample   = int(inf["nsample"])
    strategy  = inf["threshold_strategy"]

    df     = pd.read_csv(csv_path).head(eval_rows)
    labels = df[label_col].values
    df_feat = df.drop(columns=[label_col, "timestamp"], errors="ignore")

    results = perform_anomaly_analysis_with_diffusion(
        df=df_feat,
        nsample=nsample,
        threshold_strategy=strategy,
        model_path=inf.get("model_path") or None,
        config_path=inf.get("config_path") or None,
    )

    f1 = float(f1_score(labels, results["Anomaly"].values, zero_division=0))
    (out / "metrics.json").write_text(json.dumps({"f1_score": f1}))
    print(f"nsample={nsample}  strategy={strategy}  F1={f1:.4f}")


if __name__ == "__main__":
    main()
