# NV-Tesseract Forecasting

Tesseract Forecasting Model that learns universal representations from diverse temporal data using self-supervised pretraining for forecasting.

## Quick Start with UV

### 1. Install UV (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use pip
pip install uv
```

### 2. Set Up the Project

```bash
cd forecasting/
uv sync --group dev
uv pip install -e .  # Install package in editable mode for clean imports
```

### 3. Authenticate with Hugging Face

Since the model weights are in a private repository, you need to authenticate:

```bash
# Install huggingface-hub if not already installed
uv add huggingface_hub

# Authenticate (one-time setup)
huggingface-cli login
# Or set environment variable: export HUGGINGFACE_HUB_TOKEN="your_token"
```

### 4. Run Forecasting

The model weights will be automatically downloaded on first use:

```python
import pandas as pd
from sdk.forecasting import perform_forecasting, DEVICE

# Check device being used
print(f"Using device: {DEVICE}")

# Load your data
df = pd.read_csv("your_data.csv")  # Must have 'timestamp' and 'target' columns

# Perform forecasting - weights auto-downloaded if needed
results = perform_forecasting(
    df=df,
    timestamp_column="timestamp",
    target_column="target",
    forecast_horizon=72,  # Forecast 72 steps ahead
)

print(results)
```

Or run the example:

```bash
# Full example with model download (requires test data)
uv run python sdk/quick_example.py
```

## UV Commands Reference

### Package Management
```bash
uv add <package>           # Add dependency
uv add --dev <package>     # Add dev dependency
uv remove <package>        # Remove dependency
uv sync                    # Install all dependencies
uv sync --group dev        # Install with dev dependencies
```

### Running Code
```bash
uv run python script.py    # Run Python with project environment
uv run pytest             # Run tests
uv run ruff check         # Run linting
```

### Environment Management
```bash
uv venv                    # Create virtual environment
source .venv/bin/activate  # Activate environment (Unix)
.venv\Scripts\activate     # Activate environment (Windows)
```

## Project Structure

```
forecasting/
├── pyproject.toml         # Project configuration and dependencies
├── README.md             # This file
├── sdk/
│   ├── forecasting.py    # Main forecasting module (with auto-download)
│   ├── quick_example.py  # Example usage script
│   └── tests/            # Test files and datasets
├── dataset_longhorizon.py # Dataset utilities
├── model.py              # Model building utilities
├── standardizer.pkl      # Model weights (auto-downloaded on first use)
└── moment_head_512_6hr.pt
```

## Dependencies

### Core Dependencies
- `datasetsforecast>=1.0.0` - Dataset utilities for forecasting
- `joblib>=1.5.2` - Serialization for model artifacts
- `momentfm` - MOMENT foundation model (installed from GitHub main branch)
- `pandas>=2.1.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `torch>=2.0.0` - Deep learning framework
- `tqdm>=4.65.0` - Progress bars
- `huggingface_hub>=0.17.0` - For downloading model weights

### Development Dependencies
- `pytest>=9.0.2` - Testing framework
- `ruff>=0.8.0` - Linting and formatting

### Optional Dependencies
- `mac-mps` group: Optimized PyTorch for Apple Silicon

### Note on Dependencies
- `momentfm` is installed directly from GitHub (commit `38f7310a`) since version 0.1.5 is not yet released on PyPI
- This ensures compatibility with the latest MOMENT foundation model features

## Configuration

The project uses `pyproject.toml` for configuration:

- **Build system**: Hatchling
- **Linting**: Ruff with comprehensive rules
- **Python version**: >=3.10
- **Package management**: UV

## Model Weights

The forecasting model requires pre-trained weights from the Hugging Face repository:
- Repository: `nvidia/nv-tesseract-forecasting`
- Required files (auto-downloaded to current directory):
  - `standardizer.pkl` - Data normalization parameters
  - `moment_head_512_6hr.pt` - Model checkpoint for 6-hour forecasting

## Usage Examples

### Basic Forecasting
```python
from sdk.forecasting import perform_forecasting, DEVICE
import pandas as pd
import numpy as np

# Your data should have timestamp and target columns
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
    'target': np.random.randn(1000).cumsum()
})

# Perform forecasting - model weights auto-downloaded on first run
results = perform_forecasting(
    df=df, 
    forecast_horizon=24,
    timestamp_column="timestamp",
    target_column="target"
)
```

### Context-Enhanced Forecasting (DARR Mode)
```python
from sdk.forecasting import perform_forecasting

# With additional context data for improved accuracy
context_df = pd.read_csv("historical_data.csv")

results = perform_forecasting(
    df=df,
    context_df=context_df,  # Additional context for better predictions
    forecast_horizon=72,
    alpha=0.01,  # Blending parameter for DARR mode
    timestamp_column="timestamp",
    target_column="target"
)
```

### Manual Weight Paths (Optional)
```python
from sdk.forecasting import perform_forecasting

# If you want to specify custom paths for model weights
results = perform_forecasting(
    df=df,
    forecast_horizon=72,
    standardizer_pkl="custom/path/standardizer.pkl",  # Default: "standardizer.pkl"
    ckpt="custom/path/model_checkpoint.pt"            # Default: "moment_head_512_6hr.pt"
)
```

## Development

### Running Tests
```bash
uv run pytest
```

### Linting
```bash
uv run ruff check
uv run ruff format
```

### Adding Dependencies
```bash
# Add runtime dependency
uv add new-package

# Add development dependency
uv add --dev pytest-cov
```

## Troubleshooting

### UV Installation Issues
If UV installation fails:
1. Try manual installation: https://docs.astral.sh/uv/getting-started/installation/
2. Use pip: `pip install uv`
3. Use pipx: `pipx install uv`

### Dependency Resolution Issues
If you see errors about `momentfm>=0.1.5` not being available:
- This is expected - we install `momentfm` directly from GitHub since v0.1.5 isn't on PyPI yet
- The `pyproject.toml` is configured to handle this automatically
- Make sure `tool.hatch.metadata.allow-direct-references = true` is set

### Model Weight Download Issues
1. Ensure you're authenticated: `huggingface-cli login`
2. Verify repository access to `nvidia/nv-tesseract-forecasting`
3. Check network connectivity

### Import Errors
Make sure to run Python through UV to use the project environment:
```bash
uv run python your_script.py
```

## License

[Add your license information here]