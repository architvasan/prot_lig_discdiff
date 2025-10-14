#!/usr/bin/env python3
"""
Debug script to identify training issues with progress bar and sampling.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_progress_bar():
    """Test tqdm progress bar updates."""
    print("🧪 Testing progress bar updates...")
    
    try:
        from tqdm import tqdm
        import time
        
        # Create a simple progress bar test
        data = range(10)
        pbar = tqdm(data, desc="Test Progress")
        
        for i in pbar:
            # Simulate some work
            time.sleep(0.1)
            
            # Update progress bar with metrics
            loss = 1.0 / (i + 1)  # Decreasing loss
            accuracy = i / 10.0   # Increasing accuracy
            
            postfix_dict = {
                'loss': f"{loss:.4f}",
                'acc': f"{accuracy:.3f}",
                'step': i
            }
            
            pbar.set_postfix(postfix_dict)
        
        print("✅ Progress bar test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Progress bar test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading."""
    print("🧪 Testing configuration loading...")
    
    try:
        # Test loading the protein config
        config_path = "configs/config_protein.yaml"
        if not Path(config_path).exists():
            print(f"⚠️  Config file not found: {config_path}")
            return False
        
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        print(f"✅ Config loaded successfully")
        print(f"   Sampling config: {config_dict.get('sampling', 'Not found')}")
        print(f"   Training config: {config_dict.get('training', 'Not found')}")
        
        # Check specific sampling settings
        sampling_config = config_dict.get('sampling', {})
        print(f"   Sample interval: {sampling_config.get('sample_interval', 'Not set')}")
        print(f"   Save to file: {sampling_config.get('save_to_file', 'Not set')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sampling_function():
    """Test the sampling function directly."""
    print("🧪 Testing sampling function...")
    
    try:
        from protlig_ddiff.sampling.protein_sampling import sample_during_training
        
        # Create minimal mock components
        class MockModel:
            def eval(self): pass
            def __call__(self, x, sigma, use_subs=True):
                batch_size, seq_len = x.shape
                vocab_size = 26
                return torch.randn(batch_size, seq_len, vocab_size)
        
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
        
        if sequences is not None and len(sequences) > 0:
            print(f"✅ Sampling function returned {len(sequences)} sequences")
            return True
        else:
            print(f"⚠️  Sampling function returned: {sequences}")
            return False
        
    except Exception as e:
        print(f"❌ Sampling function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_training_script_issues():
    """Check for common issues in the training script."""
    print("🧪 Checking training script for common issues...")
    
    try:
        script_path = "protlig_ddiff/train/run_train_clean.py"
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        issues = []
        
        # Check for progress bar update location
        if "pbar.set_postfix" in content:
            print("✅ Progress bar update found")
        else:
            issues.append("Progress bar update not found")
        
        # Check for sampling call
        if "self.sample_sequences" in content:
            print("✅ Sampling call found")
        else:
            issues.append("Sampling call not found")
        
        # Check for step increment logic
        if "self.current_step += 1" in content:
            print("✅ Step increment found")
        else:
            issues.append("Step increment not found")
        
        # Check for metrics update
        if "self.metrics.update" in content:
            print("✅ Metrics update found")
        else:
            issues.append("Metrics update not found")
        
        if issues:
            print(f"⚠️  Found {len(issues)} potential issues:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print("✅ No obvious issues found in training script")
            return True
        
    except Exception as e:
        print(f"❌ Training script check failed: {e}")
        return False

def main():
    """Run all debug tests."""
    print("🔍 Running training debug tests...\n")
    
    tests = [
        test_progress_bar,
        test_config_loading,
        test_sampling_function,
        check_training_script_issues,
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
    
    print(f"📊 Debug Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All debug tests passed!")
        print("\n💡 Suggestions:")
        print("   1. Check that your config file has the correct sampling settings")
        print("   2. Verify that the model is training (loss should decrease)")
        print("   3. Make sure you're running with the correct device settings")
        return 0
    else:
        print("💥 Some debug tests failed!")
        print("\n💡 Common fixes:")
        print("   1. Check import paths and dependencies")
        print("   2. Verify config file exists and is valid")
        print("   3. Check device compatibility (CPU vs GPU)")
        return 1

if __name__ == "__main__":
    exit(main())
