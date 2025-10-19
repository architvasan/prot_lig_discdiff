#!/usr/bin/env python3
"""
Test script to check the actual model size based on config_protein.yaml
"""

import torch
import yaml
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_simple_namespace(config_dict):
    """Convert nested dict to simple namespace for compatibility."""
    class SimpleNamespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    # Create nested namespaces
    result = SimpleNamespace()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            setattr(result, key, SimpleNamespace(**value))
        else:
            setattr(result, key, value)
    
    return result

def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_number(num):
    """Format large numbers in human-readable format."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def estimate_memory_usage(total_params, dtype=torch.float32):
    """Estimate memory usage for model parameters."""
    bytes_per_param = 4 if dtype == torch.float32 else 2  # 4 bytes for float32, 2 for float16
    param_memory_gb = (total_params * bytes_per_param) / (1024**3)
    
    # Rough estimates for training (includes gradients, optimizer states, activations)
    training_memory_gb = param_memory_gb * 4  # Conservative estimate
    
    return param_memory_gb, training_memory_gb

def main():
    print("🔍 Testing Model Size from config_protein.yaml")
    print("=" * 60)
    
    # Load configuration
    config_path = "configs/config_protein.yaml"
    try:
        config_dict = load_config(config_path)
        config = create_simple_namespace(config_dict)
        print(f"✅ Loaded config from: {config_path}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    # Extract model parameters
    model_config = config.model
    print(f"\n📊 Model Configuration:")
    print(f"   Dimension: {model_config.dim}")
    print(f"   Heads: {model_config.n_heads}")
    print(f"   Layers: {model_config.n_layers}")
    print(f"   Condition dim: {model_config.cond_dim}")
    print(f"   Dropout: {getattr(model_config, 'dropout', 0.0)}")
    print(f"   MLP ratio: {getattr(model_config, 'mlp_ratio', 4)}")
    
    # Try to import and create the model
    try:
        from protlig_ddiff.models.transformer_v100 import DiscDiffModel
        
        print(f"\n🏗️  Creating model...")
        
        # Create model with config
        model = DiscDiffModel(config)
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        
        print(f"\n📈 Model Size Analysis:")
        print(f"   Total parameters: {format_number(total_params)} ({total_params:,})")
        print(f"   Trainable parameters: {format_number(trainable_params)} ({trainable_params:,})")
        
        # Memory estimates
        param_memory, training_memory = estimate_memory_usage(total_params)
        print(f"\n💾 Memory Estimates:")
        print(f"   Model parameters: {param_memory:.2f} GB")
        print(f"   Training memory (est): {training_memory:.2f} GB")
        print(f"   Per-rank memory (240 ranks): {training_memory/240:.3f} GB")
        
        # Compare with expected sizes
        print(f"\n🎯 Size Comparison:")
        if total_params > 300e6:
            print(f"   ✅ Large model (>300M params) - Good for complex protein modeling")
        elif total_params > 100e6:
            print(f"   ✅ Medium model (100-300M params) - Balanced size")
        else:
            print(f"   ⚠️  Small model (<100M params) - May have limited capacity")
        
        # Architecture breakdown
        print(f"\n🔧 Architecture Breakdown:")
        for name, module in model.named_children():
            if hasattr(module, 'parameters'):
                module_params = sum(p.numel() for p in module.parameters())
                percentage = (module_params / total_params) * 100
                print(f"   {name}: {format_number(module_params)} ({percentage:.1f}%)")
        
        print(f"\n✅ Model size test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Error importing model: {e}")
        print("   Make sure you're in the correct directory and dependencies are installed")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
