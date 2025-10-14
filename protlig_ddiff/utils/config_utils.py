"""
Configuration utilities for training setup and hyperparameter management.
"""
import os
import json
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any


def safe_getattr(obj, path, default=None):
    """Safely get nested attributes with default fallback."""
    try:
        parts = path.split('.')
        current = obj
        for part in parts:
            current = getattr(current, part)
        return current
    except AttributeError:
        return default


def load_config(config_file):
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config


def create_namespace_from_dict(config_dict):
    """Convert a dictionary to a namespace object for attribute access."""
    class ConfigNamespace:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigNamespace(**value))
                else:
                    setattr(self, key, value)
        
        def __repr__(self):
            items = []
            for key, value in self.__dict__.items():
                if isinstance(value, ConfigNamespace):
                    items.append(f"{key}=<ConfigNamespace>")
                else:
                    items.append(f"{key}={value}")
            return f"ConfigNamespace({', '.join(items)})"
    
    return ConfigNamespace(**config_dict)


def get_vocab_size_from_config(config):
    """Extract vocabulary size from configuration with fallbacks."""
    # Try different possible locations for vocab size
    vocab_size_paths = [
        'tokens',
        'vocab_size', 
        'data.vocab_size_protein',
        'model.vocab_size',
        'data.vocab_size',
        'tokenizer.vocab_size'
    ]
    
    for path in vocab_size_paths:
        vocab_size = safe_getattr(config, path)
        if vocab_size is not None:
            return int(vocab_size)
    
    # Default fallback
    print("⚠️  No vocab_size found in config, using default: 33")
    return 33


def get_model_config_from_config(config):
    """Extract model configuration parameters."""
    model_config = {
        'dim': int(safe_getattr(config, 'model.dim', 768)),
        'n_heads': int(safe_getattr(config, 'model.n_heads', 12)),
        'n_layers': int(safe_getattr(config, 'model.n_layers', 12)),
        'vocab_size': int(get_vocab_size_from_config(config)),
        'max_seq_len': int(safe_getattr(config, 'model.max_seq_len', 512)),
        'cond_dim': int(safe_getattr(config, 'model.cond_dim', 128)),
        'scale_by_sigma': bool(safe_getattr(config, 'model.scale_by_sigma', False)),
    }
    
    return model_config


def get_training_config_from_config(config):
    """Extract training configuration parameters."""
    training_config = {
        'batch_size': int(safe_getattr(config, 'training.batch_size', 32)),
        'learning_rate': float(safe_getattr(config, 'training.learning_rate', 1e-4)),
        'weight_decay': float(safe_getattr(config, 'training.weight_decay', 0.01)),
        'warmup_steps': int(safe_getattr(config, 'training.warmup_steps', 1000)),
        'max_steps': int(safe_getattr(config, 'training.max_steps', 100000)),
        'gradient_clip_norm': float(safe_getattr(config, 'training.gradient_clip_norm', 1.0)),
        'accumulate_grad_batches': int(safe_getattr(config, 'training.accumulate_grad_batches', 1)),
        'use_ema': bool(safe_getattr(config, 'training.use_ema', True)),
        'ema_decay': float(safe_getattr(config, 'training.ema_decay', 0.9999)),
        'use_subs_loss': bool(safe_getattr(config, 'training.use_subs_loss', True)),
    }
    
    return training_config


def get_data_config_from_config(config):
    """Extract data configuration parameters."""
    data_config = {
        'max_length': int(safe_getattr(config, 'data.max_length', 512)),
        'tokenize_on_fly': bool(safe_getattr(config, 'data.tokenize_on_fly', False)),
        'use_streaming': bool(safe_getattr(config, 'data.use_streaming', False)),
        'num_workers': int(safe_getattr(config, 'data.num_workers', 4)),
        'pin_memory': bool(safe_getattr(config, 'data.pin_memory', True)),
    }
    
    return data_config


def get_noise_config_from_config(config):
    """Extract noise schedule configuration."""
    noise_config = {
        'type': str(safe_getattr(config, 'noise.type', 'loglinear')),
        'eps': float(safe_getattr(config, 'noise.eps', 1e-3)),
        'sigma_min': float(safe_getattr(config, 'noise.sigma_min', 1e-4)),
        'sigma_max': float(safe_getattr(config, 'noise.sigma_max', 1.0)),
    }
    
    return noise_config


def get_curriculum_config_from_config(config):
    """Extract curriculum learning configuration."""
    curriculum_config = {
        'enabled': bool(safe_getattr(config, 'curriculum.enabled', True)),
        'start_bias': float(safe_getattr(config, 'curriculum.start_bias', 0.8)),
        'end_bias': float(safe_getattr(config, 'curriculum.end_bias', 0.0)),
        'decay_steps': int(safe_getattr(config, 'curriculum.decay_steps', 10000)),
    }
    
    return curriculum_config


def save_config(config, save_path):
    """Save configuration to file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert namespace back to dict if needed
    if hasattr(config, '__dict__'):
        config_dict = {}
        for key, value in config.__dict__.items():
            if hasattr(value, '__dict__'):
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value
    else:
        config_dict = config
    
    with open(save_path, 'w') as f:
        if save_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif save_path.suffix.lower() == '.json':
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported save format: {save_path.suffix}")


def print_config_summary(config):
    """Print a summary of the configuration."""
    print("\n" + "="*60)
    print("📋 CONFIGURATION SUMMARY")
    print("="*60)
    
    # Model config
    model_config = get_model_config_from_config(config)
    print("🏗️  Model Configuration:")
    for key, value in model_config.items():
        print(f"   {key}: {value}")
    
    # Training config
    training_config = get_training_config_from_config(config)
    print("\n🚀 Training Configuration:")
    for key, value in training_config.items():
        print(f"   {key}: {value}")
    
    # Data config
    data_config = get_data_config_from_config(config)
    print("\n📊 Data Configuration:")
    for key, value in data_config.items():
        print(f"   {key}: {value}")
    
    print("="*60 + "\n")
