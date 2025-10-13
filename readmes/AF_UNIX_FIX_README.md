# Fix for AF_UNIX Path Too Long Error on Polaris

## Problem Description

When running the training script on Polaris, you may encounter this error:

```
OSError: AF_UNIX path too long
```

This error occurs because:

1. **Long temporary directory paths**: Polaris uses very long paths for temporary directories
2. **Unix socket limitations**: Unix domain sockets have a path length limit (typically 108 characters)
3. **Python multiprocessing**: PyTorch's DataLoader with `num_workers > 0` uses multiprocessing, which creates Unix sockets in the temporary directory

## Solution Implemented

We've implemented a comprehensive fix that addresses this issue at multiple levels:

### 1. Automatic Temporary Directory Setup

The training script now automatically sets up a shorter temporary directory path:

- **Primary locations tried**: `/tmp`, `/dev/shm`, current directory
- **Fallback**: Creates `tmp_pytorch` in current directory
- **Environment variables set**: `TMPDIR`, `TEMP`, `TMP`

### 2. Multiprocessing Configuration

- **Start method**: Set to `spawn` for better HPC compatibility
- **Fallback mechanism**: Automatically reduces `num_workers` if AF_UNIX error occurs
- **Worker reduction sequence**: Original → Half → 1 → 0 (single-threaded)

### 3. DDP Integration

The fix is integrated into the distributed training setup:

- Applied in both Aurora and Polaris DDP setup functions
- Ensures consistent temporary directory across all processes

## Files Modified

### Core Training Script
- `protlig_ddiff/train/run_train_clean.py`
  - Added `setup_temp_directory()` function
  - Added multiprocessing start method configuration
  - Added DataLoader fallback mechanism
  - Added cleanup on exit

### DDP Utilities
- `protlig_ddiff/utils/ddp_utils.py`
  - Added `setup_temp_directory_if_needed()` function
  - Integrated into both Aurora and Polaris setup functions

### Run Scripts
- `run_polaris.sh`
  - Added environment variable setup in PBS script
  - Added environment variable setup for interactive runs
  - Added cleanup on completion

## Additional Tools

### Test Script
- `test_unix_socket_fix.py`
  - Tests multiprocessing functionality
  - Tests PyTorch DataLoader with different worker counts
  - Verifies the fix works correctly

### Environment Setup Script
- `fix_polaris_tmpdir.sh`
  - Standalone script to set up environment variables
  - Can be sourced before running training
  - Includes automatic cleanup

## Usage

### Option 1: Use Modified Training Script (Recommended)

The fix is automatically applied when you run the training script:

```bash
python protlig_ddiff/train/run_train_clean.py \
    --config config.yaml \
    --datafile data.pt \
    --cluster polaris
```

### Option 2: Use Environment Setup Script

```bash
# Source the environment setup
source fix_polaris_tmpdir.sh

# Run your training
python protlig_ddiff/train/run_train_clean.py \
    --config config.yaml \
    --datafile data.pt \
    --cluster polaris
```

### Option 3: Manual Environment Setup

```bash
# Set temporary directory manually
export TMPDIR="/tmp/pytorch_$$"
mkdir -p $TMPDIR
export TEMP=$TMPDIR
export TMP=$TMPDIR

# Additional settings
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Run training
python protlig_ddiff/train/run_train_clean.py \
    --config config.yaml \
    --datafile data.pt \
    --cluster polaris
```

## Testing the Fix

Run the test script to verify the fix works:

```bash
python test_unix_socket_fix.py
```

This will test:
- Basic multiprocessing functionality
- PyTorch DataLoader with multiple workers
- Automatic fallback mechanisms

## Troubleshooting

### If you still get AF_UNIX errors:

1. **Check path length**: Ensure your working directory path isn't extremely long
2. **Reduce workers**: Manually set `num_workers=0` in your config
3. **Use /dev/shm**: If available, manually set `TMPDIR=/dev/shm/pytorch_$$`

### If training is slow:

- The fix may reduce `num_workers` to 0, making data loading single-threaded
- This is a trade-off for stability on HPC systems
- Consider using pre-processed data to minimize loading overhead

### For debugging:

- The training script prints which temporary directory is being used
- It also prints if `num_workers` is reduced due to AF_UNIX errors
- Check the logs for these messages

## Technical Details

### Why This Happens

Unix domain sockets are used by Python's multiprocessing module for inter-process communication. These sockets are created as files in the temporary directory, and their full path must be under the system limit (usually 108 characters).

### Why Our Fix Works

1. **Shorter paths**: We use `/tmp` or `/dev/shm` which are much shorter than Polaris default temp dirs
2. **Graceful degradation**: If multiprocessing fails, we fall back to single-threaded data loading
3. **Early setup**: We set the temp directory before any multiprocessing operations begin

### Performance Impact

- **Minimal**: The fix primarily affects initialization, not training performance
- **Data loading**: May be slower if `num_workers` is reduced, but training will work
- **Memory**: Using `/dev/shm` (if available) can actually improve performance as it's RAM-based

## Support

If you continue to experience issues after applying this fix, please:

1. Run the test script and share the output
2. Check your specific Polaris configuration
3. Consider using single-threaded data loading (`num_workers=0`) as a workaround
