#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for multi-GPU interpretability — Pass A (Jacobian) + sharded Pass B (Shapley/coupling).

The same logic is exposed through the SDK via ``perform_forecasting(..., interpretability=True)``
with ``interpretability_shapley_workers``, ``interpretability_parallel_passes``, and
``interpretability_devices``.  This script is a thin benchmark / demo entry point — prefer
the SDK for production use (JSON, PDF, coupling CSVs).

Example (2+ CUDA GPUs required for ``--benchmark``):

    uv run python run_interpretability_sharded.py \\
        --csv sdk/tests/datasets/ETTh_4feature.csv \\
        --shapley-workers 2 \\
        --parallel-passes \\
        --benchmark

    # checkpoint head size vs explanation length are independent:
    uv run python run_interpretability_sharded.py \\
        --model-horizon 72 --forecast-horizon 100 \\
        --shapley-workers 2 --benchmark

Single-GPU / MPS: omit ``--benchmark``; Pass A + B run sequentially.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dataset_longhorizon import Standardizer
from interpretability_parallel import (
    InterpretabilityPassConfig,
    build_pass_configs,
    run_interpretability_passes,
)
from parallel import parse_devices_arg
from sdk.forecasting import (
    DEFAULT_BACKBONE_NAME,
    DEFAULT_CROSS_CHANNEL_CHECKPOINT_NAME,
    download_model_weights,
)

_DEFAULT_CSV = (
    Path(__file__).resolve().parent / "sdk" / "tests" / "datasets" / "ETTh_4feature.csv"
)


def _load_standardizer(path: str) -> Standardizer:
    artifact = joblib.load(path)
    if isinstance(artifact, Standardizer):
        return artifact
    if isinstance(artifact, dict):
        return Standardizer(mean=artifact["mean"], std=artifact["std"])
    raise TypeError(f"Unsupported standardizer artifact: {type(artifact)!r}")


def _load_context_window(
    csv_path: Path,
    *,
    seq_len: int,
    standardizer: Standardizer,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Load trailing window; use all numeric columns (same as SDK)."""
    df = pd.read_csv(csv_path)
    value_cols = [c for c in df.columns if c != "timestamp"]
    if not value_cols:
        raise ValueError(f"{csv_path}: no feature columns besides timestamp")
    tail = df[value_cols].tail(seq_len)
    if len(tail) < seq_len:
        raise ValueError(f"{csv_path}: need at least {seq_len} rows, got {len(tail)}")
    values_lc = tail.to_numpy(dtype=np.float32)
    c_data = int(values_lc.shape[1])
    c_std = int(standardizer.mean.shape[0])
    if c_std < c_data:
        print(
            f"[warn] standardizer C={c_std} < data C={c_data}; "
            "broadcasting the saved mean/std across all columns (same as SDK).",
            flush=True,
        )
    series_lc = standardizer.transform(values_lc)
    x_ct = series_lc.T.copy()
    mask_l = np.ones((seq_len,), dtype=np.int64)
    return x_ct, mask_l, value_cols, c_data


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", type=Path, default=_DEFAULT_CSV,
                   help="Input CSV with a timestamp column and one or more feature columns")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument(
        "--model-horizon", type=int, default=72,
        help="Native checkpoint head size (must match the .pt file; default 72 for run8_best_model_cr.pt)",
    )
    p.add_argument(
        "--forecast-horizon", type=int, default=100,
        help="Explanation / Pass B extension length (may exceed model_horizon)",
    )
    p.add_argument("--n-lags", type=int, default=128)
    p.add_argument("--coupling-transitions", type=int, default=128,
                   help="Number of Shapley transitions for Pass B (more = slower but more coupling coverage)")
    p.add_argument("--shapley-n-samples", type=int, default=32,
                   help="KernelSHAP samples per transition")
    p.add_argument("--standardizer", default="standardizer.pkl")
    p.add_argument("--ckpt", default=DEFAULT_CROSS_CHANNEL_CHECKPOINT_NAME)
    p.add_argument("--model-name", default=DEFAULT_BACKBONE_NAME)
    p.add_argument("--devices", default="auto",
                   help="Device spec: auto | cpu | cuda:0 | cuda:0,cuda:1 | ...")
    p.add_argument("--shapley-workers", type=int, default=0,
                   help="Shard Pass B across N GPUs (0 = off; >=2 activates sharding)")
    p.add_argument("--parallel-passes", action="store_true",
                   help="Run Pass A (Jacobian, GPU 0) in parallel with sharded Pass B")
    p.add_argument("--transition-batch", type=int, default=8,
                   help="Stack N transitions into one GPU forward pass (amortises kernel launch)")
    p.add_argument("--chan-batch-size", type=int, default=64)
    p.add_argument(
        "--benchmark", action="store_true",
        help="Compare serial Pass B vs multi-GPU sharded Pass B (requires 2+ CUDA devices)",
    )
    args = p.parse_args()

    csv_path = args.csv.expanduser().resolve()
    if not csv_path.exists():
        print(f"[err] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    std_path, ckpt_path = download_model_weights(
        standardizer_pkl=args.standardizer,
        ckpt=args.ckpt,
    )
    standardizer = _load_standardizer(std_path)
    seq_len = int(args.seq_len)
    model_horizon = int(args.model_horizon)
    forecast_horizon = int(args.forecast_horizon)
    x_ct, input_mask_l, _labels, c_use = _load_context_window(
        csv_path, seq_len=seq_len, standardizer=standardizer
    )

    devices = parse_devices_arg(str(args.devices))
    n_cuda = sum(1 for d in devices if d.type == "cuda")
    print(f"[data] {csv_path.name}  C={c_use}  L={seq_len}  devices={devices}", flush=True)

    pass_cfg = InterpretabilityPassConfig(
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        model_horizon=model_horizon,
        n_lags=int(args.n_lags),
        coupling_transitions=int(args.coupling_transitions),
        shapley_n_samples=int(args.shapley_n_samples),
        transition_batch=int(args.transition_batch),
        channel_batch_size=int(args.chan_batch_size),
        devices=str(args.devices),
        parallel_passes=bool(args.parallel_passes),
        shapley_workers=int(args.shapley_workers),
    )
    ckpt_resolved = str(Path(ckpt_path).resolve())

    if args.benchmark:
        if n_cuda < 2:
            print(
                "[err] --benchmark compares serial vs multi-GPU sharding and needs "
                f"2+ CUDA devices (got {devices}).",
                file=sys.stderr,
            )
            sys.exit(1)
        if int(args.shapley_workers) < 2:
            print("[err] --benchmark requires --shapley-workers >= 2", file=sys.stderr)
            sys.exit(1)

        from interpretability_parallel import _run_pass_b_serial, _WorkerInit

        _, chan_cfg_sh, series_ext, mask_ext, _ = build_pass_configs(x_ct, cfg=pass_cfg)
        init = _WorkerInit(
            ckpt_path=ckpt_resolved,
            model_name=str(args.model_name),
            seq_len=seq_len,
            model_horizon=model_horizon,
            use_cross_channel=True,
            cross_channel_heads=8,
            cross_channel_dropout=0.1,
        )
        model = init(0, devices[0])["model"]

        print("\n=== Serial Pass B ===", flush=True)
        report_s, t_s = _run_pass_b_serial(
            model,
            series_ext=series_ext,
            seq_len=seq_len,
            mask_ext=mask_ext,
            chan_cfg_sh=chan_cfg_sh,
            device=devices[0],
        )
        print(f"  {t_s:.1f}s  norm={report_s.coupling_off_diag_norm:.4e}", flush=True)

        cfg_parallel = replace(
            pass_cfg,
            shapley_workers=int(args.shapley_workers),
            parallel_passes=True,
        )
        print(f"\n=== Parallel (Pass A + {args.shapley_workers}-shard Pass B) ===", flush=True)
        t0 = time.time()
        result = run_interpretability_passes(
            x_ct=x_ct,
            input_mask_l=input_mask_l,
            cfg=cfg_parallel,
            use_channel_axis=True,
            ckpt_path=ckpt_resolved,
            model_name=str(args.model_name),
            use_cross_channel=True,
        )
        wall = time.time() - t0
        norm = result.coupling_report.coupling_off_diag_norm if result.coupling_report else None
        print(
            f"  wall {wall:.1f}s  pass B {result.pass_b_seconds:.1f}s  norm={norm}",
            flush=True,
        )
        speedup = t_s / max(result.pass_b_seconds, 1e-6)
        print(f"\n[benchmark] serial Pass B / sharded Pass B: {speedup:.2f}x", flush=True)
        return

    result = run_interpretability_passes(
        x_ct=x_ct,
        input_mask_l=input_mask_l,
        cfg=pass_cfg,
        use_channel_axis=True,
        ckpt_path=ckpt_resolved,
        model_name=str(args.model_name),
        use_cross_channel=True,
    )
    if result.coupling_report is not None:
        print(
            f"coupling_off_diag_norm={result.coupling_report.coupling_off_diag_norm:.4e}  "
            f"parallel={result.parallel}",
            flush=True,
        )


if __name__ == "__main__":
    main()
