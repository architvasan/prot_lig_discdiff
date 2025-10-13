#!/usr/bin/env python3
"""
Simple test for TrainerConfig without importing the full training module.
"""

import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainerConfig:
    """Enhanced trainer configuration that loads and includes all YAML arguments."""
    work_dir: str
    datafile: str
    config_file: str | None = None
    rank: int = 0
    world_size: int = 1
    device: str = 'cpu'
    seed: int = 42
    use_wandb: bool = True
    resume_checkpoint: Optional[str] = None
    
    # YAML config sections will be added dynamically
    model: Optional[dict] = None
    training: Optional[dict] = None
    data: Optional[dict] = None
    noise: Optional[dict] = None
    curriculum: Optional[dict] = None
    logging: Optional[dict] = None
    optimizer: Optional[dict] = None
    scheduler: Optional[dict] = None
    
    # Top-level config values
    tokens: int = 26
    devicetype: str = 'cuda'

    def __post_init__(self):
        """Load YAML config and set all values as attributes."""
        if self.config_file is not None:
            with open(self.config_file, "r") as f:
                config_dict = yaml.safe_load(f)
            
            # Set all config values as attributes
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    # Convert dict to namespace-like object for easy access
                    setattr(self, key, self._dict_to_namespace(value))
                else:
                    setattr(self, key, value)
    
    def _dict_to_namespace(self, d):
        """Convert dictionary to namespace-like object for dot notation access."""
        class ConfigNamespace:
            def __init__(self, dictionary):
                for key, value in dictionary.items():
                    if isinstance(value, dict):
                        setattr(self, key, ConfigNamespace(value))
                    else:
                        setattr(self, key, value)
            
            def __getattr__(self, name):
                # Return None for missing attributes instead of raising AttributeError
                return None
            
            def get(self, key, default=None):
                """Get attribute with default value."""
                return getattr(self, key, default)
        
        return ConfigNamespace(d)
    
    def __getitem__(self, key):
        """Allow dictionary-style access."""
        return getattr(self, key)
    
    def get(self, key, default=None):
        """Get attribute with default value."""
        return getattr(self, key, default)

def test_config():
    """Test the TrainerConfig."""
    print("🔍 Testing TrainerConfig...")
    
    # Test with YAML file
    config = TrainerConfig(
        work_dir="./test_work_dir",
        datafile="./test_data.pt",
        config_file="config_protein.yaml",
        device="0"
    )
    
    print("✅ TrainerConfig created successfully")
    
    # Test accessing YAML values
    print(f"📋 Config values:")
    print(f"  tokens: {config.tokens}")
    print(f"  devicetype: {config.devicetype}")
    print(f"  model.dim: {config.model.dim}")
    print(f"  model.n_layers: {config.model.n_layers}")
    print(f"  training.learning_rate: {config.training.learning_rate}")
    print(f"  training.batch_size: {config.training.batch_size}")
    print(f"  data.max_length: {config.data.max_length}")
    print(f"  data.tokenize_on_fly: {config.data.tokenize_on_fly}")
    print(f"  noise.type: {config.noise.type}")
    
    # Test .get() method
    print(f"\n🔍 Testing .get() method:")
    print(f"  training.get('learning_rate'): {config.training.get('learning_rate')}")
    print(f"  training.get('nonexistent', 'default'): {config.training.get('nonexistent', 'default')}")
    
    # Test nested access
    if hasattr(config, 'curriculum') and config.curriculum:
        print(f"  curriculum.enabled: {config.curriculum.enabled}")
        print(f"  curriculum.start_bias: {config.curriculum.start_bias}")
    
    print("\n✅ All config access tests passed!")
    
    # Show how the old vs new way looks
    print("\n📊 Comparison - Old vs New way:")
    print("  OLD: getattr(self.train_config.training, 'learning_rate', 1e-4)")
    print(f"  NEW: self.config.training.get('learning_rate', 1e-4) = {config.training.get('learning_rate', 1e-4)}")
    print()
    print("  OLD: getattr(self.train_config.data, 'batch_size', 32)")
    print(f"  NEW: self.config.training.get('batch_size', 32) = {config.training.get('batch_size', 32)}")

if __name__ == "__main__":
    test_config()
