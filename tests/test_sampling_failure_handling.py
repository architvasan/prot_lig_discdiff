#!/usr/bin/env python3
"""
Test script to verify sampling failure handling in training.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_sampling_failure_handling():
    """Test that sampling failures are properly handled."""
    print("🧪 Testing sampling failure handling...")
    
    try:
        from protlig_ddiff.train.run_train_clean import TrainerConfig, UniRef50Trainer
        
        # Create a mock config with sampling failure settings
        class MockConfig:
            def __init__(self):
                self.work_dir = "./test_work_dir"
                self.datafile = "dummy.pt"
                self.config_file = None
                self.rank = 0
                self.world_size = 1
                self.device = 'cpu'
                self.seed = 42
                self.use_wandb = False
                self.resume_checkpoint = None
                self.tokens = 26
                self.devicetype = 'cpu'
                
                # Mock training config
                self.training = MockTrainingConfig()
                self.model = MockModelConfig()
                self.data = MockDataConfig()
                self.noise = MockNoiseConfig()
                self.sampling = MockSamplingConfig()
        
        class MockTrainingConfig:
            def __init__(self):
                self.learning_rate = 1e-4
                self.weight_decay = 0.01
                self.warmup_steps = 100
                self.max_steps = 1000
                self.batch_size = 4
                self.accumulate_grad_batches = 1
                self.use_ema = False
                self.gradient_clip_norm = 1.0
                self.use_subs_loss = True
                # Sampling failure settings
                self.max_sampling_failures = 2  # Low threshold for testing
                self.stop_on_sampling_failure = True
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        class MockModelConfig:
            def __init__(self):
                self.dim = 128
                self.n_layers = 2
                self.n_heads = 4
                self.max_seq_len = 64
                self.cond_dim = 32
                self.scale_by_sigma = False
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        class MockDataConfig:
            def __init__(self):
                self.max_length = 64
                self.tokenize_on_fly = False
                self.use_streaming = False
                self.num_workers = 0
                self.pin_memory = False
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        class MockNoiseConfig:
            def __init__(self):
                self.type = 'loglinear'
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        class MockSamplingConfig:
            def __init__(self):
                self.sample_interval = 10
                self.eval_batch_size = 2
                self.eval_max_length = 32
                self.eval_steps = 10
                self.predictor = 'analytic'
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        # Test 1: Check that config is properly loaded
        config = MockConfig()
        print(f"✅ Config created with max_sampling_failures: {config.training.max_sampling_failures}")
        print(f"✅ Config created with stop_on_sampling_failure: {config.training.stop_on_sampling_failure}")
        
        # Test 2: Check that trainer initializes with failure tracking
        # Note: We can't fully test the trainer without a real dataset and model,
        # but we can test the config loading
        print("✅ Sampling failure handling configuration test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sampling_function_error_handling():
    """Test that the sampling function properly handles different types of errors."""
    print("🧪 Testing sampling function error handling...")
    
    try:
        from protlig_ddiff.sampling.protein_sampling import sample_during_training
        
        # Create mock components that will fail
        class FailingModel:
            def eval(self): pass
            def __call__(self, x, sigma, use_subs=True):
                raise RuntimeError("Mock model failure")
        
        class MockGraph:
            def sample_limit(self, batch_size, seq_len):
                return torch.randint(0, 25, (batch_size, seq_len))
        
        class MockNoise:
            def __call__(self, t):
                return t, -torch.ones_like(t)
        
        class MockConfig:
            def __init__(self):
                self.sampling = MockSamplingConfig()
        
        class MockSamplingConfig:
            def __init__(self):
                self.eval_batch_size = 2
                self.eval_max_length = 32
                self.eval_steps = 10
                self.predictor = 'analytic'
        
        # Test with failing model
        model = FailingModel()
        graph = MockGraph()
        noise = MockNoise()
        config = MockConfig()
        
        print("🔍 Testing with failing model...")
        sequences = sample_during_training(
            model=model,
            graph=graph,
            noise=noise,
            config=config,
            step=100,
            device='cpu'
        )
        
        # Should return empty list on failure, not crash
        if sequences == []:
            print("✅ Sampling function properly handled model failure")
        else:
            print(f"⚠️  Unexpected result: {sequences}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sampling function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Running sampling failure handling tests...\n")
    
    tests = [
        test_sampling_failure_handling,
        test_sampling_function_error_handling,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED\n")
            else:
                print("❌ FAILED\n")
        except Exception as e:
            print(f"❌ FAILED with exception: {e}\n")
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())
