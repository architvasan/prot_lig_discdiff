# 🎛️ Configurable Training Intervals Guide

## 📊 **All Training Intervals Now Configurable**

Previously, many intervals were hardcoded in `run_train_clean.py`. Now they're all configurable through the YAML config file!

---

## ⚙️ **Configuration Sections**

### **1. Logging Configuration**
```yaml
logging:
  log_interval: 20           # Log metrics to wandb every N steps
  debug_interval: 100        # Print debug info (sigma stats, etc.) every N steps
```

**What this controls:**
- **`log_interval`**: How often training metrics are logged to Wandb
- **`debug_interval`**: How often debug information is printed to console

### **2. Validation Configuration**
```yaml
validation:
  eval_freq: 500             # Run validation every N steps
  checkpoint_freq: 1000      # Save checkpoint every N steps
  checkpoint_on_improvement: true  # Only save if validation improves
```

**What this controls:**
- **`eval_freq`**: How often validation loss is computed
- **`checkpoint_freq`**: How often checkpoints are saved
- **`checkpoint_on_improvement`**: Whether to save only when validation improves

### **3. Sampling Configuration**
```yaml
sampling:
  sample_interval: 1000      # Generate sequences every N steps
  min_steps_before_sampling: 1000  # Wait N steps before starting sampling
  eval_batch_size: 4         # Number of sequences per rank
  eval_max_length: 256       # Max length for generated sequences
  eval_steps: 50             # Number of sampling steps
```

**What this controls:**
- **`sample_interval`**: How often sequences are generated for monitoring
- **`min_steps_before_sampling`**: Minimum steps before sampling starts
- **`eval_batch_size`**: Number of sequences generated per rank
- **`eval_max_length`**: Maximum length of generated sequences
- **`eval_steps`**: Number of diffusion steps for sampling

---

## 🚀 **Performance Optimization Settings**

### **Current Optimized Configuration:**
```yaml
# Fast logging, no performance impact
logging:
  log_interval: 20           # Frequent metrics logging
  debug_interval: 100        # Occasional debug prints

# Moderate validation frequency
validation:
  eval_freq: 500             # Validation every 500 steps
  checkpoint_freq: 1000      # Checkpoints every 1000 steps

# Reduced sampling overhead (100x reduction!)
sampling:
  sample_interval: 1000      # 10x less frequent (was 100)
  eval_batch_size: 4         # 2.5x fewer sequences (was 10)
  eval_max_length: 256       # 2x shorter (was 512)
  eval_steps: 50             # 2x fewer steps (was 100)
```

---

## 📈 **Performance Impact Analysis**

### **Before Optimization:**
- **Sampling**: Every 100 steps
- **Sequences**: 240 ranks × 10 = 2,400 sequences
- **Operations**: 2,400 × 100 × 512 = 122M token generations
- **Result**: 1.5s/it → 10s/it (6.7x slowdown)

### **After Optimization:**
- **Sampling**: Every 1000 steps (10x less frequent)
- **Sequences**: 240 ranks × 4 = 960 sequences (2.5x fewer)
- **Operations**: 960 × 50 × 256 = 12M token generations (10x fewer)
- **Result**: Expected ~2-3s/it (no more slowdowns)

### **Total Reduction:**
- **Frequency**: 10x reduction
- **Per-sampling overhead**: 10x reduction
- **Total sampling overhead**: 100x reduction

---

## 🎯 **Recommended Settings for Different Use Cases**

### **🏃 Fast Development/Debugging:**
```yaml
logging:
  log_interval: 10           # Very frequent logging
  debug_interval: 50         # Frequent debug info

validation:
  eval_freq: 100             # Frequent validation
  checkpoint_freq: 500       # Frequent checkpoints

sampling:
  sample_interval: 500       # More frequent sampling
  eval_batch_size: 2         # Fewer sequences
  eval_max_length: 128       # Shorter sequences
  eval_steps: 25             # Fewer steps
```

### **🎯 Production Training (Current):**
```yaml
logging:
  log_interval: 20           # Balanced logging
  debug_interval: 100        # Moderate debug info

validation:
  eval_freq: 500             # Balanced validation
  checkpoint_freq: 1000      # Standard checkpoints

sampling:
  sample_interval: 1000      # Reduced overhead
  eval_batch_size: 4         # Balanced sequences
  eval_max_length: 256       # Reasonable length
  eval_steps: 50             # Efficient sampling
```

### **🚀 Maximum Performance:**
```yaml
logging:
  log_interval: 50           # Less frequent logging
  debug_interval: 500        # Minimal debug info

validation:
  eval_freq: 1000            # Less frequent validation
  checkpoint_freq: 2000      # Less frequent checkpoints

sampling:
  sample_interval: 2000      # Minimal sampling
  eval_batch_size: 2         # Minimal sequences
  eval_max_length: 128       # Short sequences
  eval_steps: 25             # Fast sampling
```

---

## 🔧 **Implementation Details**

### **Files Modified:**
1. **`protlig_ddiff/train/run_train_clean.py`**:
   - Added `self.log_interval` and `self.debug_interval` from config
   - Replaced hardcoded `20` with `self.log_interval`
   - Replaced hardcoded `100` with `self.debug_interval`
   - All validation and sampling intervals already configurable

2. **`configs/config_protein.yaml`**:
   - Added `logging.debug_interval` configuration
   - Updated comments for clarity
   - Set optimized values for 240-rank training

### **Backward Compatibility:**
- All intervals have sensible defaults if not specified
- Existing configs will continue to work
- New intervals are optional

---

## 📊 **Monitoring Your Settings**

### **What to Watch:**
1. **Training speed**: Should be stable ~2-3s/it
2. **Memory usage**: Monitor per-rank memory consumption
3. **ESM perplexity trends**: Should improve continuously
4. **Validation loss**: Should decrease over time

### **Tuning Guidelines:**
- **If training is slow**: Increase `sample_interval`, reduce `eval_batch_size`
- **If you need more monitoring**: Decrease `sample_interval`, increase `eval_batch_size`
- **If logs are too noisy**: Increase `log_interval` and `debug_interval`
- **If you need more checkpoints**: Decrease `checkpoint_freq`

---

## 🎉 **Benefits**

✅ **Full control** over all training intervals  
✅ **No more hardcoded values** in the training script  
✅ **Easy performance tuning** through config changes  
✅ **100x sampling overhead reduction** for 240-rank training  
✅ **Backward compatible** with existing configurations  
✅ **Clear documentation** of what each setting controls  

Now you can easily tune your training performance without modifying code! 🚀
