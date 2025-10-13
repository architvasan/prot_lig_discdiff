# Polaris Training Script Usage Guide

## Overview

The `run_polaris.sh` script has been enhanced to support hardcoded configuration, making it much easier to use for repeated training runs. You can now set your common parameters once at the top of the script and run training with minimal command-line arguments.

## Quick Start

### 1. Configure the Script (One-time setup)

Edit the **HARDCODED SETTINGS** section at the top of `run_polaris.sh`:

```bash
# Project paths (MODIFY THESE TO YOUR ACTUAL PATHS)
PROJECT_ROOT="/eagle/YourProject/username/prot_lig_discdiff"
HARDCODED_CONFIG_FILE="${PROJECT_ROOT}/config_protein.yaml"
HARDCODED_DATA_FILE="${PROJECT_ROOT}/data/processed_uniref50.pt"
HARDCODED_WORK_DIR="${PROJECT_ROOT}/experiments/sedd_$(date +%Y%m%d_%H%M%S)"

# Job settings (MODIFY AS NEEDED)
HARDCODED_ACCOUNT="YourAllocation"  # *** SET YOUR POLARIS ACCOUNT HERE ***
HARDCODED_TIME_LIMIT=2  # Hours
HARDCODED_QUEUE="workq"

# Training settings (MODIFY AS NEEDED)
HARDCODED_WANDB_PROJECT="protein-discrete-diffusion-polaris"
HARDCODED_NODES=1
HARDCODED_PPN=4  # Processes per node (Polaris has 4 GPUs per node)
```

### 2. Run Training

Once configured, you can run training with minimal arguments:

```bash
# Submit as PBS job (recommended for production)
./run_polaris.sh --submit

# Interactive run (for testing)
./run_polaris.sh

# Override specific settings
./run_polaris.sh --data /path/to/different/data.pt --time 8 --submit
```

## Configuration Options

### Hardcoded Settings (set once in script)

| Setting | Description | Example |
|---------|-------------|---------|
| `PROJECT_ROOT` | Base directory for your project | `/eagle/YourProject/username/prot_lig_discdiff` |
| `HARDCODED_CONFIG_FILE` | Path to training config | `${PROJECT_ROOT}/config_protein.yaml` |
| `HARDCODED_DATA_FILE` | Path to training data | `${PROJECT_ROOT}/data/processed_uniref50.pt` |
| `HARDCODED_ACCOUNT` | Your Polaris allocation | `YourAllocation` |
| `HARDCODED_TIME_LIMIT` | Job time limit (hours) | `2` |
| `HARDCODED_NODES` | Number of nodes | `1` |
| `HARDCODED_PPN` | Processes per node | `4` |

### Command Line Options (override hardcoded settings)

| Option | Description | Example |
|--------|-------------|---------|
| `--data PATH` | Training data file | `--data /path/to/data.pt` |
| `--config PATH` | Config file | `--config /path/to/config.yaml` |
| `--work_dir PATH` | Working directory | `--work_dir /path/to/workdir` |
| `--account NAME` | Polaris allocation | `--account YourAllocation` |
| `--time HOURS` | Job time limit | `--time 8` |
| `--nodes NUM` | Number of nodes | `--nodes 2` |
| `--ppn NUM` | Processes per node | `--ppn 4` |
| `--wandb_project NAME` | Wandb project | `--wandb_project my-project` |
| `--wandb_name NAME` | Wandb run name | `--wandb_name my-run` |
| `--seed NUM` | Random seed | `--seed 123` |
| `--submit` | Submit as PBS job | `--submit` |

## Usage Examples

### Example 1: First-time Setup

1. Edit the script:
```bash
vim run_polaris.sh
# Set PROJECT_ROOT, HARDCODED_ACCOUNT, and file paths
```

2. Test interactively:
```bash
./run_polaris.sh
```

3. Submit production job:
```bash
./run_polaris.sh --submit
```

### Example 2: Different Datasets

```bash
# Use different data file
./run_polaris.sh --data /eagle/YourProject/data/dataset2.pt --submit

# Use different config and longer time
./run_polaris.sh --config config_large.yaml --time 12 --submit
```

### Example 3: Multi-node Training

```bash
# Use 2 nodes with 4 processes each (8 total GPUs)
./run_polaris.sh --nodes 2 --ppn 4 --time 8 --submit
```

### Example 4: Different Wandb Project

```bash
# Use different wandb project for experiment tracking
./run_polaris.sh --wandb_project protein-experiment-v2 --submit
```

## File Structure

The script expects this file structure:

```
/eagle/YourProject/username/prot_lig_discdiff/
├── run_polaris.sh                    # This script
├── config_protein.yaml               # Training configuration
├── protlig_ddiff/
│   └── train/
│       └── run_train_clean.py        # Training script
├── data/
│   └── processed_uniref50.pt         # Training data
└── experiments/                      # Output directory
    └── sedd_YYYYMMDD_HHMMSS/         # Auto-generated work dirs
```

## Monitoring Jobs

### Check Job Status
```bash
# Check all your jobs
qstat -u $USER

# Check specific job
qstat JOB_ID

# Check job details
qstat -f JOB_ID
```

### View Job Output
```bash
# Real-time monitoring
tail -f /path/to/workdir/job_output.log

# View errors
tail -f /path/to/workdir/job_error.log

# View training log
tail -f /path/to/workdir/training.log
```

## Troubleshooting

### Common Issues

1. **Account not set**:
   ```
   ❌ Account/allocation name is required for Polaris
   ```
   **Solution**: Set `HARDCODED_ACCOUNT` in the script or use `--account`

2. **Data file not found**:
   ```
   ❌ Data file not found: /path/to/data.pt
   ```
   **Solution**: Check the path in `HARDCODED_DATA_FILE` or use `--data`

3. **Config file not found**:
   ```
   ❌ Config file not found: /path/to/config.yaml
   ```
   **Solution**: Check the path in `HARDCODED_CONFIG_FILE` or use `--config`

### Debug Mode

To see what settings are being used:

```bash
./run_polaris.sh --help
```

This shows all current hardcoded settings and available options.

### Validation

The script validates:
- ✅ Required account is set
- ✅ Data file exists
- ✅ Config file exists
- ✅ Work directory can be created

## Advanced Usage

### Custom Environment Variables

You can set additional environment variables in the script:

```bash
# In the script, add to the environment setup section:
export CUSTOM_VAR="value"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
```

### Different Queues

```bash
# Use debug queue for short jobs
./run_polaris.sh --queue debug --time 1 --submit

# Use preemptable queue for longer jobs
./run_polaris.sh --queue preemptable --time 24 --submit
```

### Resource Optimization

```bash
# Single GPU for testing
./run_polaris.sh --nodes 1 --ppn 1 --submit

# Maximum resources (4 nodes, 16 GPUs)
./run_polaris.sh --nodes 4 --ppn 4 --time 12 --submit
```

## Best Practices

1. **Start Small**: Test with `--nodes 1 --ppn 1` first
2. **Use Hardcoded Settings**: Set common paths once in the script
3. **Monitor Resources**: Check GPU utilization with `nvidia-smi`
4. **Save Configs**: The script automatically copies config to work directory
5. **Use Descriptive Names**: Set meaningful `--wandb_name` for experiments
6. **Check Quotas**: Ensure you have sufficient allocation hours

## Support

For issues with the script:
1. Check the hardcoded settings at the top of the script
2. Run `./run_polaris.sh --help` to see current configuration
3. Test interactively before submitting jobs
4. Check Polaris documentation for system-specific issues
