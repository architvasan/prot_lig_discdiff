# Validation and Checkpointing System

## Overview

The training system now includes a comprehensive validation and checkpointing system that:

1. **Runs SUBS validation every 500 steps** on the validation dataset
2. **Saves checkpoints every 1000 steps** or when validation improves
3. **Implements early stopping** based on validation patience
4. **Tracks validation history** and restores state from checkpoints

## Configuration

Add the following section to your `config_protein.yaml`:

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

## Features

### 1. SUBS Validation Every 500 Steps

- **Method**: `validate_model_subs()`
- **What it does**: 
  - Runs the model on validation batches using SUBS loss
  - Computes validation metrics (perplexity, accuracy, etc.)
  - Tracks validation loss history
- **Frequency**: Configurable via `validation.eval_freq` (default: 500 steps)

### 2. Smart Checkpointing

- **Method**: `save_checkpoint()` with improved logic
- **Checkpoint conditions**:
  - Every 1000 steps (configurable via `validation.checkpoint_freq`)
  - When validation loss improves by more than `min_delta`
  - Can be configured to save only on improvement
- **Checkpoint contents**:
  ```python
  {
      'step': current_step,
      'epoch': current_epoch,
      'best_loss': best_validation_loss,
      'val_loss': current_validation_loss,
      'val_loss_history': list_of_validation_losses,
      'steps_without_improvement': count,
      'model_state_dict': model_weights,
      'optimizer_state_dict': optimizer_state,
      'scheduler_state_dict': scheduler_state,
      # ... other training state
  }
  ```

### 3. Early Stopping

- **Trigger**: When validation doesn't improve for `patience` evaluations
- **Default patience**: 10 evaluations (= 5000 steps with default eval_freq)
- **Improvement threshold**: Configurable via `min_delta` (default: 0.001)

### 4. Validation Tracking

- **Best validation loss**: Tracks the best validation loss seen so far
- **Loss history**: Maintains a list of all validation losses
- **Steps without improvement**: Counts evaluations without improvement
- **State restoration**: All tracking state is saved and restored from checkpoints

## Usage

### Training with Validation

The validation system is automatically integrated into the training loop:

```python
# In training loop (every 500 steps):
if step % self.val_eval_freq == 0:
    val_loss, val_metrics = self.validate_model_subs()
    improved = self.update_validation_tracking(val_loss)
    
    # Log to wandb
    self.log_to_wandb({
        'validation/loss': val_loss,
        'validation/best_loss': self.best_val_loss,
        **val_metrics
    }, step=step)
    
    # Check early stopping
    if self.steps_without_improvement >= self.patience:
        print("Early stopping triggered!")
        break

# Checkpointing (every 1000 steps or on improvement):
if self.should_save_checkpoint(step, val_loss):
    self.save_checkpoint(step, epoch, val_loss, is_best)
```

### Resuming from Checkpoint

When resuming training, the validation state is automatically restored:

```python
# Validation state is restored from checkpoint
self.best_val_loss = checkpoint.get('val_loss', float('inf'))
self.val_loss_history = checkpoint.get('val_loss_history', [])
self.steps_without_improvement = checkpoint.get('steps_without_improvement', 0)
```

## Monitoring

### Wandb Logging

The system logs comprehensive validation metrics to Wandb:

- `validation/loss`: Current validation loss
- `validation/best_loss`: Best validation loss so far
- `validation/steps_without_improvement`: Steps without improvement
- `validation/mean_log_prob`: Mean log probability on validation set
- `validation/perplexity`: Validation perplexity
- `validation/accuracy`: Validation accuracy
- `checkpoint/step`: Checkpoint step
- `checkpoint/is_best`: Whether this is the best checkpoint

### Console Output

```
🔍 Running SUBS validation at step 1500...
✅ Validation completed: 20 batches, avg loss: 2.3456
🎉 Validation improved: 2.4123 → 2.3456
💾 Checkpoint saved (rank 0): /path/to/checkpoint_step_1500.pth
   📊 Validation loss: 2.3456 (best: 2.3456)
```

## Best Practices

1. **Validation frequency**: 500 steps is a good balance between monitoring and training speed
2. **Checkpoint frequency**: 1000 steps ensures you don't lose too much progress
3. **Patience**: Set based on your training length (10 evaluations = ~5000 steps)
4. **Batch limit**: Limit validation batches (20) to keep validation fast
5. **Early stopping**: Enable for long training runs to prevent overfitting

## Files Modified

1. **`configs/config_protein.yaml`**: Added validation configuration section
2. **`protlig_ddiff/train/run_train_clean.py`**:
   - Added validation tracking variables in `setup_validation_tracking()`
   - Implemented `validate_model_subs()` method for SUBS validation
   - Enhanced checkpoint saving with validation info in `save_training_checkpoint()`
   - Added `load_training_checkpoint()` method for state restoration
   - Updated training loop with validation and checkpointing logic
   - Added validation loader creation in `setup_data_loaders()`

## Testing

Run the test script to verify the system:

```bash
python test_validation_checkpointing.py
```

This validates:
- Configuration loading
- Validation tracking logic
- Checkpoint decision making
- Checkpoint format with validation info
