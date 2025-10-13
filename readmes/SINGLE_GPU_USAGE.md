# Single GPU Training Guide

## Overview

You can absolutely run the training script on a single GPU without using distributed training (DDP). The script automatically detects whether to use single GPU or distributed mode based on the presence of the `--cluster` argument.

## Quick Start

### **Method 1: Direct Script Usage (Recommended)**

```bash
python protlig_ddiff/train/run_train_clean.py \
    --config config_protein.yaml \
    --datafile /path/to/your/data.pt \
    --device 0 \
    --work_dir ./single_gpu_run
```

### **Method 2: Using the Wrapper Script**

```bash
python run_single_gpu.py \
    --config config_protein.yaml \
    --datafile /path/to/your/data.pt \
    --device 0
```

## Device Specification Options

The script accepts several device formats:

| Format | Description | Example |
|--------|-------------|---------|
| `cpu` | Use CPU only | `--device cpu` |
| `0`, `1`, `2` | GPU number (assumes CUDA) | `--device 0` |
| `cuda:0` | Full CUDA specification | `--device cuda:0` |
| `xpu:0` | Intel XPU specification | `--device xpu:0` |

## Complete Examples

### **Basic Single GPU Training**

```bash
python protlig_ddiff/train/run_train_clean.py \
    --config config_protein.yaml \
    --datafile data/processed_uniref50.pt \
    --device 0 \
    --work_dir ./experiments/single_gpu_run \
    --wandb_project my-protein-project \
    --seed 42
```

### **CPU Training (for testing)**

```bash
python protlig_ddiff/train/run_train_clean.py \
    --config config_protein.yaml \
    --datafile data/small_test_data.pt \
    --device cpu \
    --work_dir ./experiments/cpu_test \
    --no_wandb
```

### **GPU Training without Wandb**

```bash
python protlig_ddiff/train/run_train_clean.py \
    --config config_protein.yaml \
    --datafile data/processed_uniref50.pt \
    --device cuda:0 \
    --work_dir ./experiments/no_wandb_run \
    --no_wandb
```

### **Resume from Checkpoint**

```bash
python protlig_ddiff/train/run_train_clean.py \
    --config config_protein.yaml \
    --datafile data/processed_uniref50.pt \
    --device 0 \
    --work_dir ./experiments/resumed_run \
    --resume_checkpoint ./experiments/previous_run/checkpoints/checkpoint_step_5000.pt
```

## Key Differences: Single GPU vs. Distributed

| Aspect | Single GPU | Distributed (DDP) |
|--------|------------|-------------------|
| **Activation** | No `--cluster` argument | `--cluster polaris` or `--cluster aurora` |
| **World Size** | Always 1 | Number of processes |
| **Data Sampling** | Regular random sampling | DistributedSampler |
| **Model Wrapping** | Raw model | DDP-wrapped model |
| **Synchronization** | None needed | Barriers and all-reduce |
| **Workers** | Can use more workers | Limited workers to avoid hangs |

## Performance Considerations

### **Advantages of Single GPU**

1. **Simpler setup**: No MPI or distributed configuration needed
2. **More workers**: Can use more DataLoader workers without hang issues
3. **Faster iteration**: No synchronization overhead
4. **Easier debugging**: Simpler error messages and stack traces

### **When to Use Single GPU**

- **Development and testing**: Faster iteration cycles
- **Small datasets**: When data fits comfortably on one GPU
- **Prototyping**: When experimenting with model architectures
- **Limited resources**: When you only have access to one GPU

### **Performance Tips for Single GPU**

1. **Increase batch size**: Use larger batches to maximize GPU utilization
2. **Use more workers**: Set `num_workers: 4-8` in your config
3. **Enable pin_memory**: Set `pin_memory: true` in your config
4. **Mixed precision**: Consider using automatic mixed precision (AMP)

## Configuration Adjustments

### **Recommended Config Changes for Single GPU**

```yaml
# In your config_protein.yaml
data:
  num_workers: 4          # Can use more workers on single GPU
  pin_memory: true        # Faster data transfer
  batch_size: 32          # Increase if you have memory

training:
  batch_size: 32          # Match data batch_size
  accumulate_grad_batches: 4  # Simulate larger batch size
  use_ema: true           # EMA works well on single GPU
```

### **Memory Optimization**

If you run out of memory:

```yaml
data:
  batch_size: 16          # Reduce batch size
  
training:
  batch_size: 16
  accumulate_grad_batches: 8  # Maintain effective batch size of 128
  gradient_clip_norm: 1.0     # Prevent gradient explosion
```

## Monitoring Single GPU Training

### **Check GPU Usage**

```bash
# Monitor GPU utilization
nvidia-smi -l 1

# Check memory usage
watch -n 1 nvidia-smi
```

### **Training Logs**

The script provides clear indicators for single GPU mode:

```
🔧 Environment setup: device=cuda:0, seed=42
🏗️  Model setup: 50,000,000 parameters
📊 Data setup: 1000 batches per epoch
🚀 Training components initialized
📊 Gradient accumulation: 4 batches
```

Notice the absence of DDP-related messages like:
- "DDP: Hi from rank 0 of 4..."
- "✅ DDP initialized..."
- "🔄 Rank 0: Ready for training epoch"

## Troubleshooting

### **Common Issues**

1. **CUDA out of memory**:
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce batch size or use gradient accumulation

2. **GPU not found**:
   ```
   ⚠️  GPU 0 not available, using GPU 0
   ```
   **Solution**: Check `nvidia-smi` and use correct GPU index

3. **Slow data loading**:
   ```
   🔧 Reduced num_workers to 0 to avoid multiprocessing issues
   ```
   **Solution**: Check your data file path and format

### **Debug Commands**

```bash
# Test with minimal config
python protlig_ddiff/train/run_train_clean.py \
    --config config_protein.yaml \
    --datafile data/small_test.pt \
    --device 0 \
    --work_dir ./debug_run \
    --no_wandb

# Test data loading only
python debug_hangs.py

# Check available GPUs
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

## Wrapper Script Features

The `run_single_gpu.py` wrapper provides additional convenience:

### **Features**

- **Automatic device validation**: Checks if specified GPU exists
- **Auto-generated run names**: Creates timestamped run names
- **Environment setup**: Sets `CUDA_VISIBLE_DEVICES` automatically
- **Input validation**: Checks if config and data files exist
- **Simplified interface**: Fewer required arguments

### **Usage**

```bash
# Minimal usage
python run_single_gpu.py --config config.yaml --datafile data.pt

# With options
python run_single_gpu.py \
    --config config.yaml \
    --datafile data.pt \
    --device 1 \
    --work_dir ./my_experiment \
    --wandb_project my-project \
    --no_wandb
```

## Best Practices

1. **Start small**: Test with a small dataset first
2. **Monitor resources**: Keep an eye on GPU memory and utilization
3. **Use checkpointing**: Save checkpoints regularly for long runs
4. **Validate data**: Ensure your data loads correctly before long training
5. **Experiment tracking**: Use wandb or disable it consistently

## Comparison with Distributed Training

| Use Case | Single GPU | Distributed |
|----------|------------|-------------|
| **Development** | ✅ Recommended | ❌ Overkill |
| **Small datasets** | ✅ Perfect | ❌ Unnecessary |
| **Large datasets** | ⚠️ May be slow | ✅ Recommended |
| **Production training** | ⚠️ Limited scale | ✅ Better throughput |
| **Debugging** | ✅ Easier | ❌ More complex |

The single GPU mode is perfect for development, testing, and smaller-scale training runs!
