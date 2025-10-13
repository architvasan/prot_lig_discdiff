# Fix for Training Hangs on Polaris

## Problem Description

After fixing the AF_UNIX path too long error, you may experience hanging issues during training. Common hang points include:

1. **DataLoader iteration** - Multiprocessing workers getting stuck
2. **Wandb initialization** - Network timeouts during setup
3. **Distributed training synchronization** - Processes waiting for each other
4. **Progress bar updates** - tqdm hanging in distributed environments
5. **File I/O operations** - Slow filesystem access

## Solution Implemented

We've implemented comprehensive hang detection and prevention mechanisms:

### 🔧 **Core Fixes in Training Script**

#### 1. **DataLoader Hang Prevention**
- **Timeout protection**: Added 30-second timeout for worker processes
- **Progressive fallback**: Automatically reduces `num_workers` if issues occur
- **Test-before-use**: Tests DataLoader with a sample batch before training
- **Persistent workers disabled**: Prevents worker process accumulation

#### 2. **Distributed Training Robustness**
- **Barrier timeouts**: All distributed barriers have 30-60 second timeouts
- **Error isolation**: Individual process failures don't crash entire job
- **Progress logging**: Regular heartbeat messages to detect hangs
- **Graceful degradation**: Continues with fewer workers if needed

#### 3. **Wandb Hang Prevention**
- **Setup timeout**: 60-second timeout for wandb initialization
- **Offline fallback**: Automatically disables wandb if setup fails
- **Error handling**: Continues training even if logging fails

#### 4. **Training Loop Robustness**
- **Epoch-level error handling**: Recovers from individual epoch failures
- **Consecutive failure detection**: Stops after 3 consecutive failures
- **Timeout monitoring**: Logs progress every 30 seconds
- **Graceful cleanup**: Proper resource cleanup even on failures

### 🛠️ **Additional Tools**

#### 1. **Hang Detection Script** (`debug_hangs.py`)
Tests all components that commonly cause hangs:
```bash
python debug_hangs.py
```

Tests include:
- Basic imports (torch, wandb, mpi4py)
- PyTorch multiprocessing
- MPI functionality
- Distributed initialization
- Wandb setup
- Data loading operations

#### 2. **Training with Hang Detection** (`run_training_with_hang_detection.py`)
Wrapper script that monitors training and automatically kills hung processes:
```bash
python run_training_with_hang_detection.py protlig_ddiff/train/run_train_clean.py \
    --config config.yaml --datafile data.pt --cluster polaris
```

Features:
- **Automatic hang detection**: Kills process if no output for 10 minutes
- **Environment setup**: Automatically configures optimal settings
- **Progress monitoring**: Resets timer on any output
- **Graceful termination**: Tries SIGTERM before SIGKILL

## Usage Instructions

### Option 1: Use Enhanced Training Script (Recommended)

The training script now has built-in hang prevention:

```bash
python protlig_ddiff/train/run_train_clean.py \
    --config config.yaml \
    --datafile data.pt \
    --cluster polaris \
    --no_wandb  # Disable wandb if network issues
```

### Option 2: Use Hang Detection Wrapper

For extra protection, use the hang detection wrapper:

```bash
# Set custom timeout (default: 600 seconds)
export HANG_TIMEOUT=900

python run_training_with_hang_detection.py \
    protlig_ddiff/train/run_train_clean.py \
    --config config.yaml \
    --datafile data.pt \
    --cluster polaris
```

### Option 3: Debug First

If you're experiencing hangs, run the debug script first:

```bash
python debug_hangs.py
```

This will identify which components are causing issues.

## Configuration Recommendations

### For Polaris Specifically:

1. **Reduce workers**: Set `num_workers: 0` or `num_workers: 1` in your config
2. **Disable wandb**: Use `--no_wandb` flag if network is unreliable
3. **Use shorter paths**: Ensure your working directory path isn't too long
4. **Limit batch size**: Reduce batch size if memory issues cause hangs

### Environment Variables:

```bash
# Set in your job script or before running
export TMPDIR="/tmp/pytorch_$$"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export WANDB_SILENT=true
export HANG_TIMEOUT=900  # 15 minutes
```

## Updated Run Scripts

The `run_polaris.sh` script has been updated with hang prevention:

```bash
./run_polaris.sh --config config.yaml --datafile data.pt --submit
```

## Troubleshooting

### If Training Still Hangs:

1. **Run debug script**:
   ```bash
   python debug_hangs.py
   ```

2. **Check specific components**:
   - If DataLoader test fails: Use `num_workers=0`
   - If MPI test fails: Check MPI configuration
   - If Wandb test fails: Use `--no_wandb`

3. **Use minimal configuration**:
   ```bash
   python protlig_ddiff/train/run_train_clean.py \
       --config config.yaml \
       --datafile data.pt \
       --cluster polaris \
       --no_wandb
   ```

4. **Monitor with timeout**:
   ```bash
   timeout 3600 python protlig_ddiff/train/run_train_clean.py \
       --config config.yaml --datafile data.pt --cluster polaris
   ```

### Common Hang Locations and Solutions:

| Hang Location | Symptoms | Solution |
|---------------|----------|----------|
| DataLoader creation | Hangs after "Setting up data loaders" | Set `num_workers=0` |
| First batch iteration | Hangs at "Starting epoch 1" | Use hang detection wrapper |
| Wandb setup | Hangs after "Setting up Wandb" | Use `--no_wandb` |
| DDP initialization | Hangs after "DDP: Hi from rank..." | Check network/MPI config |
| Progress bar | Hangs during training loop | Disable tqdm in config |

## Performance Impact

The hang prevention measures have minimal performance impact:

- **DataLoader**: May reduce to single-threaded loading (slower I/O, but stable)
- **Monitoring**: Adds ~1% overhead for progress logging
- **Timeouts**: No impact unless hangs occur
- **Error handling**: Minimal overhead for exception handling

## Monitoring Training

The enhanced training script provides better logging:

```
🔧 Set temporary directory to: /tmp/pytorch_12345
🔧 Limited num_workers to 2 for distributed training
🔧 Trying DataLoader with 2 workers...
🧪 Testing DataLoader...
✅ DataLoader test successful with 2 workers
🚀 Starting epoch 1, step 0
🔄 Rank 0: Processing batch 100, step 50
✅ Epoch 1 completed in 120.5s | Avg Loss: 2.3456
```

Look for these messages to ensure training is progressing normally.

## Support

If hangs persist after applying these fixes:

1. Share the output of `python debug_hangs.py`
2. Include the last 50 lines of training output before the hang
3. Specify your Polaris job configuration (nodes, tasks per node, etc.)
4. Try the minimal configuration with `num_workers=0` and `--no_wandb`
