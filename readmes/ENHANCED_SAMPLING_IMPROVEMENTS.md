# Enhanced Sampling Improvements

This document outlines the three major improvements made to the protein sequence sampling system.

## 🎯 **Summary of Improvements**

### **1. ESM Perplexity Evaluation** ✅
- **What**: Calculate ESM (Evolutionary Scale Modeling) perplexity for each generated sequence
- **Why**: ESM perplexity provides a measure of how "protein-like" a sequence is according to a state-of-the-art protein language model
- **Implementation**: New `ESMEvaluator` class with masked language modeling approach

### **2. Rank-Specific Generation** ✅
- **What**: Ensure different sequences are generated on each rank in distributed training
- **Why**: Previously all ranks were generating identical sequences, reducing diversity
- **Implementation**: Rank-specific random seeds and enhanced sampling function

### **3. Increased Sequence Count** ✅
- **What**: Generate 10 sequences per rank instead of 4
- **Why**: More sequences provide better statistics and monitoring
- **Implementation**: Updated configuration and sampling parameters

---

## 📁 **Files Modified**

### **New Files Created:**
1. **`protlig_ddiff/utils/esm_evaluator.py`** - ESM perplexity calculation utilities
2. **`test_enhanced_sampling.py`** - Test script for all improvements
3. **`ENHANCED_SAMPLING_IMPROVEMENTS.md`** - This documentation

### **Files Modified:**
1. **`configs/config_protein.yaml`** - Updated sampling configuration
2. **`protlig_ddiff/sampling/protein_sampling.py`** - Enhanced sampling function
3. **`protlig_ddiff/train/run_train_clean.py`** - Updated training script integration

---

## 🔧 **Configuration Changes**

### **Updated `configs/config_protein.yaml`:**

```yaml
# Sampling configuration for monitoring during training
sampling:
  sample_interval: 100       # Sample every 100 steps
  min_steps_before_sampling: 100  # Wait this many steps before starting sampling
  eval_batch_size: 10        # Number of sequences to sample per rank (increased from 4 to 10)
  eval_max_length: 512       # Max length for evaluation samples
  eval_steps: 100            # Number of sampling steps
  predictor: "euler"         # Sampling predictor
  save_to_file: true         # Save sampled sequences to file
  calculate_esm_perplexity: true  # Calculate ESM perplexity for generated sequences
  esm_model: "esm2_t6_8M_UR50D"   # ESM model for perplexity calculation
  esm_batch_size: 4          # Batch size for ESM evaluation
```

---

## 🔬 **ESM Perplexity Evaluation**

### **Features:**
- **Masked Language Modeling**: Uses ESM's MLM approach for robust perplexity calculation
- **Multiple Masks**: Averages over 5 different random masks per sequence
- **Batch Processing**: Efficient batch processing with configurable batch size
- **Error Handling**: Graceful handling of invalid sequences and CUDA OOM

### **Usage:**
```python
from protlig_ddiff.utils.esm_evaluator import calculate_esm_perplexity

sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
perplexities = calculate_esm_perplexity(
    sequences=sequences,
    model_name="esm2_t6_8M_UR50D",
    batch_size=4,
    mask_fraction=0.15,
    seed=42
)
```

### **ESM Models Available:**
- `esm2_t6_8M_UR50D` (8M parameters, fastest)
- `esm2_t12_35M_UR50D` (35M parameters, balanced)
- `esm2_t30_150M_UR50D` (150M parameters, most accurate)

---

## 🎯 **Rank-Specific Generation**

### **Problem Solved:**
Previously, all ranks in distributed training were generating identical sequences because they used the same random seed.

### **Solution:**
```python
# Rank-specific seed generation
rank_seed = config.seed + step * 1000 + rank * 10000

# Set rank-specific random state
torch.manual_seed(rank_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(rank_seed)
```

### **Benefits:**
- ✅ **Different sequences per rank**: Each rank generates unique sequences
- ✅ **Better diversity**: More varied sequences for analysis
- ✅ **Reproducible**: Still deterministic given the same step and rank
- ✅ **Scalable**: Works with any number of ranks

---

## 📊 **Enhanced Logging and Monitoring**

### **New Wandb Metrics:**
```python
log_data = {
    "samples/sequences": sample_text,
    "samples/step": step,
    "samples/epoch": current_epoch,
    "samples/count": len(sequences),
    "samples/rank": sampling_rank,
    "samples/esm_perplexity_mean": np.mean(valid_perplexities),
    "samples/esm_perplexity_std": np.std(valid_perplexities),
    "samples/esm_perplexity_min": np.min(valid_perplexities),
    "samples/esm_perplexity_max": np.max(valid_perplexities),
}
```

### **Enhanced Console Output:**
```
🧬 Rank 0: Generated 10 sequences at step 1000:
  Rank 0 Sample 1: MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG... (ESM PPL: 12.34)
  Rank 0 Sample 2: ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY... (ESM PPL: 45.67)
  Rank 0 Sample 3: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA... (ESM PPL: 123.45)
```

---

## 🧪 **Testing**

### **Run Tests:**
```bash
python test_enhanced_sampling.py
```

### **Test Coverage:**
1. **Configuration Updates**: Verify config changes are loaded correctly
2. **Rank-Specific Generation**: Test that different ranks generate different sequences
3. **ESM Evaluator**: Test ESM perplexity calculation with various sequences

---

## 🚀 **Usage in Training**

### **Automatic Integration:**
The improvements are automatically integrated into your training pipeline. When you run training:

1. **Every 100 steps** (configurable), each rank will:
   - Generate **10 unique sequences** (rank-specific)
   - Calculate **ESM perplexity** for each sequence
   - Log results to **Wandb** with detailed metrics
   - Save sequences to **file** with perplexity information

### **Expected Output:**
```
🧬 Rank 0: Sampling 10 sequences (step 1000)
🔬 Rank 0: Calculating ESM perplexity for 10 sequences...
✅ Rank 0: ESM perplexity calculated. Mean: 15.67
🧬 Rank 0: Generated 10 sequences at step 1000:
  Rank 0 Sample 1: MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG... (ESM PPL: 12.34)
  ...
```

---

## 📈 **Benefits**

### **1. Better Quality Assessment**
- **ESM perplexity** provides objective measure of sequence quality
- **Lower perplexity** = more protein-like sequences
- **Track improvement** over training steps

### **2. Enhanced Diversity**
- **10 sequences per rank** instead of 4
- **Rank-specific generation** ensures variety
- **Better statistics** for monitoring

### **3. Improved Monitoring**
- **Detailed Wandb logs** with perplexity statistics
- **Rank information** for debugging distributed training
- **Enhanced console output** for real-time monitoring

---

## 🔧 **Installation Requirements**

### **For ESM Perplexity:**
```bash
pip install fair-esm
```

### **Optional: For faster ESM inference:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🎉 **Ready to Use!**

All improvements are now integrated and ready for use. Simply run your training as usual and enjoy:

- ✅ **10 sequences per rank** with rank-specific diversity
- ✅ **ESM perplexity evaluation** for quality assessment  
- ✅ **Enhanced logging** with detailed metrics
- ✅ **Better monitoring** of sequence generation quality

The system will automatically handle ESM model loading, perplexity calculation, and enhanced logging without any additional configuration needed!
