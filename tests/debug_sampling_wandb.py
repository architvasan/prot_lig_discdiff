#!/usr/bin/env python3
"""
Debug script to test sampling and wandb logging independently.
"""

import torch
import yaml
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config_loading():
    """Test that config loading works correctly."""
    print("🔍 Testing config loading...")

    # Load config
    with open("configs/config_protein.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

    print(f"Config keys: {list(config_dict.keys())}")
    print(f"Sampling config: {config_dict.get('sampling', 'NOT FOUND')}")

    # Create a simple config namespace like the TrainerConfig does
    class ConfigNamespace:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigNamespace(value))
                else:
                    setattr(self, key, value)

        def get(self, key, default=None):
            return getattr(self, key, default)

    config = ConfigNamespace(config_dict)

    print(f"Config sampling: {getattr(config, 'sampling', 'NOT FOUND')}")
    if hasattr(config, 'sampling'):
        sampling = config.sampling
        print(f"  sample_interval: {getattr(sampling, 'sample_interval', 'NOT FOUND')}")
        print(f"  eval_batch_size: {getattr(sampling, 'eval_batch_size', 'NOT FOUND')}")

    return config

def test_wandb_setup():
    """Test wandb setup."""
    print("\n🔍 Testing wandb setup...")
    
    try:
        from protlig_ddiff.utils.training_utils import setup_wandb, log_metrics
        
        # Test config
        test_config = {
            'model': {'dim': 512},
            'training': {'lr': 1e-4},
            'test': True
        }
        
        # Setup wandb
        wandb_run = setup_wandb("test-project", "debug-run", test_config)
        
        if wandb_run:
            print("✅ Wandb setup successful")
            
            # Test logging
            test_metrics = {
                'loss': 2.5,
                'accuracy': 0.3,
                'step': 100
            }
            
            log_metrics(test_metrics, step=100, wandb_run=wandb_run)
            print("✅ Wandb logging successful")
            
            # Test sequence logging
            wandb_run.log({
                "samples/sequences": "Sample 1: MKVLWAALLVTFLAG...\nSample 2: MKKLLFAIPLVVPFN...",
                "samples/step": 100
            }, step=100)
            print("✅ Wandb sequence logging successful")
            
            wandb_run.finish()
            return True
        else:
            print("❌ Wandb setup failed")
            return False
            
    except Exception as e:
        print(f"❌ Wandb test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sampling_function():
    """Test the sampling function with mock components."""
    print("\n🔍 Testing sampling function...")
    
    try:
        from protlig_ddiff.sampling.protein_sampling import sample_during_training
        
        # Create mock components
        class MockModel:
            def eval(self): pass
            def __call__(self, x, sigma, use_subs=True):
                batch_size, seq_len = x.shape
                vocab_size = 26
                return torch.randn(batch_size, seq_len, vocab_size)
        
        class MockGraph:
            def sample_limit(self, batch_size, seq_len):
                return torch.randint(0, 25, (batch_size, seq_len))
            def staggered_score(self, score, sigma):
                return torch.softmax(score, dim=-1)
            def transp_transition(self, x, sigma):
                batch_size, seq_len = x.shape
                vocab_size = 26
                return torch.ones(batch_size, seq_len, vocab_size)
        
        class MockNoise:
            def __call__(self, t):
                return t, -torch.ones_like(t)
        
        # Create mock config
        class MockConfig:
            def __init__(self):
                self.sampling = MockSamplingConfig()
        
        class MockSamplingConfig:
            def __init__(self):
                self.eval_batch_size = 2
                self.eval_max_length = 64
                self.eval_steps = 10
                self.predictor = 'analytic'
        
        # Test sampling
        model = MockModel()
        graph = MockGraph()
        noise = MockNoise()
        config = MockConfig()
        
        print("🔍 Calling sample_during_training...")
        sequences = sample_during_training(
            model=model,
            graph=graph,
            noise=noise,
            config=config,
            step=100,
            device='cpu'
        )
        
        print(f"✅ Sampling returned {len(sequences)} sequences")
        return len(sequences) > 0
        
    except Exception as e:
        print(f"❌ Sampling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 DEBUGGING SAMPLING AND WANDB")
    print("=" * 50)
    
    # Test 1: Config loading
    config = test_config_loading()
    
    # Test 2: WandB setup
    wandb_success = test_wandb_setup()
    
    # Test 3: Sampling function
    sampling_success = test_sampling_function()
    
    print("\n📊 RESULTS:")
    print(f"  Config loading: ✅")
    print(f"  WandB setup: {'✅' if wandb_success else '❌'}")
    print(f"  Sampling function: {'✅' if sampling_success else '❌'}")
    
    if wandb_success and sampling_success:
        print("\n🎉 All tests passed! The issue might be in the training loop integration.")
        print("\n💡 Suggestions:")
        print("  1. Check if sampling is being called at the right intervals")
        print("  2. Check if wandb_run is properly set in the trainer")
        print("  3. Check if the config object has the sampling section")
    else:
        print("\n❌ Some tests failed. Fix these issues first.")

if __name__ == "__main__":
    main()
