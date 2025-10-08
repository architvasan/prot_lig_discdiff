# prot_lig_discdiff
Model platform for protein/ligand discrete diffusion using different schemes

This directory contains a cleaned and reorganized version of the discrete diffusion training pipeline. 

## 📁 Package Structure

The scripts have been reorganized into the `protlig_ddiff` package with the following structure:

```
sedd_scripts/protlig_ddiff/
├── __init__.py                    # Main package init with key exports
├── models/
│   ├── __init__.py               # Model exports
│   ├── transformer_v100.py      # DiscDiffModel (renamed from SEDD)
│   └── rotary.py                 # Rotary positional embeddings
├── processing/
│   ├── __init__.py               # Processing exports
│   ├── graph_lib.py              # Graph processing utilities
│   ├── noise_lib.py              # Noise scheduling
│   └── subs_loss.py              # SUBS loss implementation
├── tests/
│   ├── __init__.py               # Test exports
│   ├── test_subs_integration.py  # Integration tests
│   ├── test_subs_loss_only.py    # Loss-specific tests
│   ├── debug_subs_detailed.py    # Detailed debugging
│   ├── debug_subs_loss.py        # Loss debugging
│   ├── debug_shapes.py           # Shape debugging
│   └── debug_normalization.py    # Normalization debugging
├── train/
│   ├── __init__.py               # Training exports
│   ├── run_train_clean.py        # Clean training script
│   └── run_train_uniref_ddp_aurora.py  # DDP training script
└── utils/
    ├── __init__.py               # Utility exports
    ├── config_utils.py           # Configuration utilities
    ├── data_utils.py             # Data loading utilities
    ├── ddp_utils.py              # Distributed training utilities
    └── training_utils.py         # Training utilities
```

### Package Components

#### 🏗️ **Models** (`protlig_ddiff.models`)
- **`DiscDiffModel`** - V100-compatible transformer with SUBS support
- **`Rotary`** - Rotary positional embeddings

#### ⚙️ **Processing** (`protlig_ddiff.processing`)
- **`subs_loss`** - SUBS loss implementation for stable training
- **`Absorbing`** - Graph utilities for absorbing state diffusion
- **`LogLinearNoise`** - Noise schedule implementations

#### 🧪 **Tests** (`protlig_ddiff.tests`)
- **Integration tests** - End-to-end SUBS integration verification
- **Debug scripts** - Detailed debugging and analysis tools
- **Unit tests** - Component-specific testing

#### 🚀 **Training** (`protlig_ddiff.train`)
- **`run_train_clean.py`** - Clean, modular training script (~300 lines)
- **`run_train_uniref_ddp_aurora.py`** - DDP training for clusters

#### 🔧 **Utilities** (`protlig_ddiff.utils`)
- **`config_utils`** - Configuration loading and parsing
- **`data_utils`** - Data loading and tokenization
- **`ddp_utils`** - Distributed training setup
- **`training_utils`** - Training optimization and monitoring

## 🚀 Quick Start

### Package Import Examples

```python
# Import the main model
from protlig_ddiff.models.transformer_v100 import DiscDiffModel

# Import SUBS loss
from protlig_ddiff.processing.subs_loss import subs_loss

# Import utilities
from protlig_ddiff.utils.config_utils import load_config
from protlig_ddiff.utils.data_utils import UniRef50Dataset

# Import from main package (convenience)
from protlig_ddiff import DiscDiffModel, subs_loss, Absorbing
```

### 1. Basic Training
```bash
python protlig_ddiff/train/run_train_clean.py \
    --config config_example.yaml \
    --datafile /path/to/your/data.jsonl \
    --work_dir ./experiments/run1 \
    --wandb_project "discrete-diffusion-training" \
    --wandb_name "experiment-1"
```

### 2. Aurora Cluster Training
```bash
python protlig_ddiff/train/run_train_clean.py \
    --config config_example.yaml \
    --datafile /path/to/your/data.jsonl \
    --work_dir ./experiments/aurora_run \
    --cluster aurora \
    --device xpu:0
```

### 3. Polaris Cluster Training
```bash
python protlig_ddiff/train/run_train_clean.py \
    --config config_example.yaml \
    --datafile /path/to/your/data.jsonl \
    --work_dir ./experiments/polaris_run \
    --cluster polaris \
    --device cuda:0
```

### 4. Running Tests
```bash
# Run integration tests
python protlig_ddiff/tests/test_subs_integration.py

# Run debug scripts
python protlig_ddiff/tests/debug_subs_detailed.py

# Test imports
python -c "from protlig_ddiff import DiscDiffModel; print('✅ Import successful')"
```

## ⚙️ Configuration

The training script uses YAML configuration files. See `config_example.yaml` for all available options.

### Key Configuration Sections:

#### Model Configuration
```yaml
model:
  dim: 768                    # Model dimension
  n_heads: 12                 # Number of attention heads
  n_layers: 12                # Number of layers
  max_seq_len: 512           # Maximum sequence length
```

#### Training Configuration
```yaml
training:
  batch_size: 32             # Training batch size
  learning_rate: 1e-4        # Learning rate
  use_subs_loss: true        # Use SUBS loss (recommended)
  use_ema: true              # Use exponential moving average
```

#### Vocabulary Configuration
```yaml
tokens: 26  # Total tokens including absorbing state
            # For 25 base tokens, use 26 (25 + 1 absorbing)
```

## 🧬 SUBS Loss Integration

The clean training script fully supports SUBS (Substitution) loss, which provides:

- **Better stability** than score-based entropy loss
- **Faster convergence** and cleaner gradients
- **Simpler implementation** similar to masked language modeling

To use SUBS loss, set `training.use_subs_loss: true` in your config.

## 📊 Data Format

The training script supports multiple data formats:

### JSONL Format (Recommended)
```json
{"sequence": "ACDEFGHIKLMNPQRSTVWY", "id": "protein_1"}
{"sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", "id": "protein_2"}
```

### Pre-tokenized Format
```json
{"tokens": [1, 5, 8, 12, 3, 9, 2], "id": "protein_1"}
{"tokens": [1, 15, 8, 7, 12, 3, 2], "id": "protein_2"}
```

## 🔧 Advanced Features

### Curriculum Learning
Automatically enabled by default. Starts with bias towards low noise levels and gradually moves to uniform sampling:

```yaml
curriculum:
  enabled: true
  start_bias: 0.8      # Start with 80% bias towards low noise
  end_bias: 0.0        # End with uniform sampling
  decay_steps: 10000   # Decay over 10k steps
```

### Gradient Accumulation
Simulate larger batch sizes by accumulating gradients across multiple mini-batches:

```yaml
training:
  batch_size: 32             # Mini-batch size
  accumulate_grad_batches: 4 # Accumulate 4 batches = effective batch size of 128
```

This is especially useful for:
- **Large models** that don't fit with large batch sizes
- **Memory-constrained environments** (Aurora XPUs, smaller GPUs)
- **Maintaining training stability** with larger effective batch sizes

### Exponential Moving Average (EMA)
Maintains a moving average of model parameters for more stable inference:

```yaml
training:
  use_ema: true
  ema_decay: 0.9999
```

### Distributed Training
Supports both Aurora (Intel XPU) and Polaris (NVIDIA GPU) clusters with automatic DDP setup.

## 📈 Monitoring

### Weights & Biases Integration
Automatic logging of:
- Training loss and accuracy
- Learning rate and gradient norms
- Model perplexity
- Training speed metrics

### Local Logging
All metrics are also saved locally in the work directory for offline analysis.

## 🔄 Checkpointing

- Automatic checkpointing every 5000 steps
- Resume training with `--resume_checkpoint path/to/checkpoint.pt`
- EMA model state included in checkpoints

## 📦 Package Benefits

### Modular Architecture
- **Separation of concerns** - Each module has a specific responsibility
- **Easy testing** - Individual components can be tested in isolation
- **Reusable components** - Utilities can be imported and used elsewhere
- **Clear dependencies** - Import structure shows component relationships

### Import Structure
```python
# Clean, organized imports
from protlig_ddiff.models import DiscDiffModel
from protlig_ddiff.processing import subs_loss, Absorbing
from protlig_ddiff.utils import load_config, UniRef50Dataset
```

### Testing Framework
```bash
# Comprehensive test suite
python protlig_ddiff/tests/test_subs_integration.py     # Integration tests
python protlig_ddiff/tests/test_subs_loss_only.py       # Unit tests
python protlig_ddiff/tests/debug_subs_detailed.py       # Debug analysis
```

## 🆚 Comparison with Original Script

| Aspect | Original Script | Reorganized Package |
|--------|----------------|---------------------|
| **Lines of Code** | 3046 lines | ~300 lines main + modular utilities |
| **Organization** | Monolithic single file | Structured package with submodules |
| **Imports** | Scattered, unclear dependencies | Clean package-based imports |
| **Testing** | Hard to test individual components | Comprehensive test suite |
| **Reusability** | Low - everything coupled | High - modular components |
| **Maintainability** | Difficult to modify | Easy to extend and modify |
| **Documentation** | Minimal inline comments | Package structure + docstrings |

## 🐛 Troubleshooting

### Common Issues

1. **Vocab Size Mismatch**
   - Ensure `tokens` in config = base_vocab_size + 1
   - Use `Absorbing(base_vocab_size)` for graph initialization

2. **Memory Issues**
   - Reduce `batch_size` in config
   - Enable `use_streaming: true` for large datasets
   - Reduce `model.dim` or `model.n_layers`

3. **DDP Issues**
   - Ensure MPI is available for Aurora/Polaris
   - Check that all processes can communicate
   - Verify correct cluster type (`--cluster aurora` or `--cluster polaris`)

### Getting Help

1. Check the configuration file format
2. Verify data file format and accessibility
3. Ensure all dependencies are installed
4. Check device availability (XPU for Aurora, CUDA for Polaris)

## 📝 Migration from Original Script

To migrate from the original scattered scripts to the new package structure:

### 1. Update Import Statements
```python
# Old imports
from transformer_v100 import SEDD
from subs_loss import subs_loss
from graph_lib import Absorbing

# New package imports
from protlig_ddiff.models.transformer_v100 import DiscDiffModel  # Note: SEDD → DiscDiffModel
from protlig_ddiff.processing.subs_loss import subs_loss
from protlig_ddiff.processing.graph_lib import Absorbing
```

### 2. Update Training Script Paths
```bash
# Old command
python run_train_clean.py --config config.yaml

# New command
python protlig_ddiff/train/run_train_clean.py --config config.yaml
```

### 3. Model Name Change
The model class has been renamed for clarity:
- **Old**: `SEDD` (confusing name)
- **New**: `DiscDiffModel` (clear, descriptive name)

### 4. Package Installation
If using the package in other projects:
```python
# Add to your Python path or install as package
import sys
sys.path.append('/path/to/sedd_scripts')
from protlig_ddiff import DiscDiffModel, subs_loss
```

### 5. Configuration Compatibility
- **Config files**: No changes needed - same YAML format
- **Data format**: Fully compatible with existing datasets
- **Checkpoints**: Compatible with existing model checkpoints
- **Vocab size**: Same `tokens: base_vocab_size + 1` format

The reorganized package maintains **full backward compatibility** while providing a much cleaner, more maintainable structure.
