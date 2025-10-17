# Data Splits and Validation System Guide

## Overview

This guide documents the complete train/validation/test splitting and validation system implemented in `run_train_clean.py`. The system provides configurable data splits, validation-based checkpointing, and comprehensive evaluation.

## Features

### ✅ **Configurable Data Splits**
- **Default ratios**: 80% train, 10% validation, 10% test
- **Fully configurable** via YAML config
- **Reproducible splits** with configurable seed
- **No overlap** between train/val/test sets

### ✅ **Validation System**
- **SUBS validation** every 500 steps (configurable)
- **Comprehensive metrics**: loss, perplexity, accuracy
- **Early stopping** based on validation patience
- **Wandb logging** of all validation metrics

### ✅ **Smart Checkpointing**
- **Save every 1000 steps** OR when validation improves
- **Validation-aware**: Only saves if improvement > min_delta
- **Best model tracking**: Automatically saves best checkpoint
- **State restoration**: Restores validation tracking when resuming

### ✅ **Test Set Evaluation**
- **Final evaluation** on test set at end of training
- **Same metrics** as validation (SUBS loss, etc.)
- **Wandb logging** of test results

## Configuration

### Data Split Configuration

Add to your `configs/config_protein.yaml`:

```yaml
# Data configuration
data:
  # ... existing data config ...
  
  # Data split ratios (must sum to 1.0)
  train_ratio: 0.8          # Training set ratio
  val_ratio: 0.1            # Validation set ratio  
  test_ratio: 0.1           # Test set ratio
  split_seed: 42            # Seed for reproducible data splits
```

### Validation Configuration

```yaml
# Validation and checkpointing
validation:
  eval_freq: 500             # Run validation every 500 steps
  checkpoint_freq: 1000      # Save checkpoint every 1000 steps
  checkpoint_on_improvement: true  # Only save checkpoint if validation improves
  patience: 10               # Number of evaluations without improvement before stopping
  min_delta: 0.001          # Minimum improvement to consider as better
  save_best_only: false     # Whether to only save the best model
  val_batch_limit: 20       # Limit validation to N batches for speed
```

## Usage

### Basic Training

```bash
python protlig_ddiff/train/run_train_clean.py \
    --config configs/config_protein.yaml \
    --datafile /path/to/your/data.txt \
    --work_dir ./work_dir
```

### Resume from Checkpoint

```bash
python protlig_ddiff/train/run_train_clean.py \
    --config configs/config_protein.yaml \
    --datafile /path/to/your/data.txt \
    --work_dir ./work_dir \
    --resume_checkpoint ./work_dir/checkpoints/checkpoint_step_5000.pt
```

### Custom Data Splits

```bash
# Use 70:20:10 split instead of default 80:10:10
# Modify config_protein.yaml:
data:
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1
```

## Implementation Details

### Data Loading Process

1. **Load full dataset** from specified data file
2. **Create reproducible splits** using `torch.utils.data.random_split`
3. **Create separate DataLoaders** for train/val/test
4. **Print split statistics** for verification

### Validation Process

1. **Every 500 steps** (configurable):
   - Switch model to eval mode
   - Run SUBS loss on validation set
   - Compute comprehensive metrics
   - Update validation tracking
   - Log to Wandb
   - Check early stopping condition

2. **Validation tracking**:
   - Track best validation loss
   - Count steps without improvement
   - Trigger early stopping if patience exceeded

### Checkpointing Process

1. **Every 1000 steps** OR **when validation improves**:
   - Save model, optimizer, scheduler state
   - Include validation tracking state
   - Save best model separately if improved
   - Log checkpoint information

2. **Checkpoint contents**:
   ```python
   checkpoint = {
       'step': current_step,
       'best_loss': best_val_loss,
       'val_loss': current_val_loss,
       'val_loss_history': val_loss_history,
       'steps_without_improvement': steps_without_improvement,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'scheduler_state_dict': scheduler.state_dict(),
       'ema_state_dict': ema_model.state_dict(),  # if EMA enabled
   }
   ```

### Test Evaluation

1. **At end of training**:
   - Run evaluation on test set
   - Use same metrics as validation
   - Log results to Wandb
   - Print final test metrics

## Files Modified

### `configs/config_protein.yaml`
- Added data split configuration section
- Enhanced validation configuration

### `protlig_ddiff/train/run_train_clean.py`
- **`create_data_splits()`**: Creates train/val/test splits
- **`setup_data_loaders()`**: Enhanced to use proper splits
- **`validate_model_subs()`**: SUBS validation on validation set
- **`evaluate_test_set()`**: Test set evaluation
- **`update_validation_tracking()`**: Validation improvement tracking
- **`should_save_checkpoint()`**: Smart checkpoint decision logic
- **`save_training_checkpoint()`**: Enhanced with validation info
- **`load_training_checkpoint()`**: Restores validation state
- **Training loop**: Integrated validation and checkpointing

## Monitoring and Logging

### Console Output
```
📊 Data splits: Train=800,000 (80.0%), Val=100,000 (10.0%), Test=100,000 (10.0%)
🔧 Data loaders created:
   📊 Train: 25,000 batches, 800,000 samples
   📊 Validation: 6,250 batches, 100,000 samples
   📊 Test: 6,250 batches, 100,000 samples

🔍 Running SUBS validation at step 500...
✅ Validation completed: 20 batches, avg loss: 2.3456
🎉 Validation improved: 2.5000 → 2.3456
💾 Checkpoint saved: ./work_dir/checkpoints/checkpoint_step_500.pt
   📊 Validation loss: 2.3456 (best: 2.3456)
```

### Wandb Metrics
- `validation/loss`: Current validation loss
- `validation/best_loss`: Best validation loss so far
- `validation/steps_without_improvement`: Steps without improvement
- `validation/perplexity`: Validation perplexity
- `validation/accuracy`: Validation accuracy
- `test/final_loss`: Final test loss
- `test/perplexity`: Final test perplexity
- `test/accuracy`: Final test accuracy

## Benefits

1. **Proper evaluation**: True validation/test splits prevent overfitting
2. **Early stopping**: Prevents overtraining and saves compute
3. **Smart checkpointing**: Only saves when model actually improves
4. **Reproducibility**: Consistent splits across runs with same seed
5. **Flexibility**: Easily configurable ratios for different use cases
6. **Monitoring**: Comprehensive logging and metrics tracking

## Best Practices

1. **Use consistent seeds** for reproducible experiments
2. **Monitor validation curves** to detect overfitting
3. **Adjust patience** based on your dataset size and training dynamics
4. **Use appropriate min_delta** to avoid saving on noise
5. **Keep test set untouched** until final evaluation
6. **Save best checkpoints** for model deployment

## Testing

Run the test suite to verify the system:

```bash
python test_data_splits.py
python test_complete_validation_system.py
```

The system is thoroughly tested and ready for production use!
