# NV-Tesseract

NVIDIA Tesseract is an open-source time series analysis library built on the [MOMENT](https://github.com/moment-research/MOMENT) foundation model, covering forecasting, anomaly detection, and classification.

## Overview

- **Forecasting**: DataFrame-first API for long-horizon time series forecasting with DARR (context-enhanced) mode.
- **Anomaly Detection** *(coming soon)*: Diffusion-based multivariate and transformer-based univariate anomaly detection.
- **Classification** *(coming soon)*: Transformer-based tabular data classification.

## Getting Started

### Installation

```bash
pip install moment-research  # MOMENT foundation model dependency
pip install pandas numpy torch
```

Then clone this repo and import directly:

```bash
git clone https://github.com/NVIDIA/NV-Tesseract.git
cd NV-Tesseract
```

### Quick Start

```python
from forecasting.sdk import perform_forecasting
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "timestamp": pd.date_range("2023-01-01", periods=600, freq="H"),
    "target": np.sin(np.linspace(0, 4 * np.pi, 600)),
    "feature_a": np.random.randn(600),
})

forecasts = perform_forecasting(
    df=df,
    seq_len=512,
    forecast_horizon=72,
    ckpt="artifacts_512_72/moment_head_512_6hr.pt",
    standardizer_pkl="artifacts_512_72/standardizer.pkl",
)
# Returns a DataFrame with `target_forecast` column containing 72 predictions
```

### DARR (Context-Enhanced) Inference

```python
darr_result = perform_forecasting(
    df=df,
    context_df=historical_df,  # Historical data for kNN retrieval
    seq_len=512,
    forecast_horizon=72,
    alpha=0.2,   # 20% direct, 80% kNN
    k=64,
    temperature=0.05,
    ckpt="artifacts_512_72/moment_head_512_6hr.pt",
    standardizer_pkl="artifacts_512_72/standardizer.pkl",
)
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- pandas, numpy
- [MOMENT](https://github.com/moment-research/MOMENT) (`AutonLab/MOMENT-1-large`)
- GPU recommended (CUDA or Apple MPS); falls back to CPU automatically

## Usage

See [`forecasting/sdk/README.md`](forecasting/sdk/README.md) for full API reference, parameter descriptions, output format, and error handling details.

See [`forecasting/sdk/quick_example.py`](forecasting/sdk/quick_example.py) for an end-to-end runnable script.

## Capabilities

| Module | Status | Description |
|--------|--------|-------------|
| `forecasting/` | Available | Time series forecasting with DARR mode |
| `ad_diffusion/` | Coming soon | Diffusion-based multivariate anomaly detection |
| `ad_transformer/` | Coming soon | Transformer-based univariate anomaly detection & classification |

## Repository Structure

```
NV-Tesseract/
├── forecasting/                 # Forecasting module (available)
│   ├── __init__.py              # Exports perform_forecasting, DEVICE
│   ├── model.py                 # Model construction utilities
│   ├── dataset_longhorizon.py   # Dataset classes for long-horizon forecasting
│   └── sdk/
│       ├── forecasting.py       # Core perform_forecasting() implementation
│       ├── quick_example.py     # End-to-end usage example
│       ├── README.md            # Full SDK API reference
│       └── tests/
│           └── datasets/        # Sample datasets for testing
├── ad_diffusion/                # Multivariate anomaly detection (coming soon)
└── ad_transformer/              # Univariate anomaly detection & classification (coming soon)
```

## Contribution Guidelines

- Start here: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`

## Security

- Vulnerability disclosure: `SECURITY.md`
- Do not file public issues for security reports.

## Support

- How to get help: [GitHub Issues](https://github.com/NVIDIA/NV-Tesseract/issues)

## License

This project is licensed under the Apache 2.0 License — see the `LICENSE` file for details.

## References

- [MOMENT: A Family of Open Time-series Foundation Models](https://github.com/moment-research/MOMENT)
