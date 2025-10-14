#!/usr/bin/env python3
"""
Debug script to identify the source of NaN values in the model.
"""

import torch
import yaml
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_model_initialization():
    """Test if the model produces NaN values right after initialization."""
    print("🧪 Testing model initialization for NaN values...")
    
    try:
        # Load config
        with open("configs/config_protein.yaml", 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create a simple config object
        class Config:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        setattr(self, key, Config(value))
                    else:
                        setattr(self, key, value)
        
        config = Config(config_dict)
        
        # Import components
        from protlig_ddiff.models.transformer_v100 import DiscDiffModel
        import protlig_ddiff.processing.graph_lib as graph_lib
        import protlig_ddiff.processing.noise_lib as noise_lib
        
        print("✅ Imports successful")
        
        # Create model
        print("🔧 Creating model...")
        model = DiscDiffModel(config)
        print("✅ Model created successfully")
        
        # Test with simple inputs
        device = torch.device('cpu')  # Use CPU for debugging
        model = model.to(device)
        model.eval()
        
        # Create test inputs
        batch_size = 2
        seq_len = 10
        vocab_size = config.tokens
        
        # Test input (protein sequence indices)
        x = torch.randint(0, vocab_size-1, (batch_size, seq_len), device=device)
        print(f"Input x shape: {x.shape}, range: [{torch.min(x)}, {torch.max(x)}]")
        
        # Test sigma (noise level)
        sigma = torch.rand(batch_size, device=device) * 0.5 + 0.1  # Range [0.1, 0.6]
        print(f"Input sigma shape: {sigma.shape}, range: [{torch.min(sigma):.4f}, {torch.max(sigma):.4f}]")
        
        # Forward pass
        print("🔧 Running forward pass...")
        with torch.no_grad():
            output = model(x, sigma, use_subs=True)
        
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{torch.min(output):.4f}, {torch.max(output):.4f}]")
        
        # Check for NaN/Inf
        if torch.any(torch.isnan(output)):
            print(f"🚨 NaN detected in output: {torch.sum(torch.isnan(output))} values")
            return False
        
        if torch.any(torch.isinf(output)):
            print(f"🚨 Inf detected in output: {torch.sum(torch.isinf(output))} values")
            return False
        
        print("✅ Model forward pass successful - no NaN/Inf values")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_noise_schedule():
    """Test the noise schedule for NaN values."""
    print("🧪 Testing noise schedule...")
    
    try:
        import protlig_ddiff.processing.noise_lib as noise_lib
        
        # Test LogLinear noise
        noise = noise_lib.LogLinearNoise()
        
        # Test various time values
        t_values = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
        
        for t in t_values:
            sigma, dsigma = noise(t.unsqueeze(0))
            print(f"t={t:.1f}: sigma={sigma.item():.6f}, dsigma={dsigma.item():.6f}")
            
            if torch.isnan(sigma) or torch.isnan(dsigma):
                print(f"🚨 NaN detected in noise schedule at t={t}")
                return False
            
            if torch.isinf(sigma) or torch.isinf(dsigma):
                print(f"🚨 Inf detected in noise schedule at t={t}")
                return False
        
        print("✅ Noise schedule test passed")
        return True
        
    except Exception as e:
        print(f"❌ Noise schedule test failed: {e}")
        return False

def test_graph_operations():
    """Test graph operations for NaN values."""
    print("🧪 Testing graph operations...")
    
    try:
        import protlig_ddiff.processing.graph_lib as graph_lib
        
        vocab_size = 26
        graph = graph_lib.Absorbing(vocab_size - 1)
        
        # Test sample_categorical with various inputs
        batch_size = 2
        seq_len = 5
        
        # Test with normal probabilities
        probs = torch.softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)
        samples = graph_lib.sample_categorical(probs)
        print(f"Normal probs test: samples shape {samples.shape}")
        
        # Test with extreme probabilities
        extreme_probs = torch.zeros(batch_size, seq_len, vocab_size)
        extreme_probs[:, :, 0] = 1000.0  # Very large logits
        extreme_probs = torch.softmax(extreme_probs, dim=-1)
        samples = graph_lib.sample_categorical(extreme_probs)
        print(f"Extreme probs test: samples shape {samples.shape}")
        
        # Test with problematic probabilities (this should trigger our fixes)
        bad_probs = torch.tensor([[[float('nan'), 0.5, 0.5], [0.3, float('inf'), 0.7]]])
        print(f"Testing with NaN/Inf probabilities...")
        samples = graph_lib.sample_categorical(bad_probs)
        print(f"Bad probs test: samples shape {samples.shape}")
        
        print("✅ Graph operations test passed")
        return True
        
    except Exception as e:
        print(f"❌ Graph operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step_simulation():
    """Simulate a training step to see where NaN might occur."""
    print("🧪 Testing training step simulation...")
    
    try:
        # Load config
        with open("configs/config_protein.yaml", 'r') as f:
            config_dict = yaml.safe_load(f)
        
        class Config:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        setattr(self, key, Config(value))
                    else:
                        setattr(self, key, value)
        
        config = Config(config_dict)
        
        # Import components
        from protlig_ddiff.models.transformer_v100 import DiscDiffModel
        import protlig_ddiff.processing.graph_lib as graph_lib
        import protlig_ddiff.processing.noise_lib as noise_lib
        from protlig_ddiff.processing.subs_loss import subs_loss_with_curriculum
        
        # Create components
        device = torch.device('cpu')
        model = DiscDiffModel(config).to(device)
        graph = graph_lib.Absorbing(config.tokens - 1)
        noise = noise_lib.LogLinearNoise()
        
        model.train()
        
        # Create training data
        batch_size = 2
        seq_len = 20
        x0 = torch.randint(0, config.tokens-1, (batch_size, seq_len), device=device)
        
        print(f"Training data x0: shape {x0.shape}, range [{torch.min(x0)}, {torch.max(x0)}]")
        
        # Sample time and noise
        t = torch.rand(batch_size, device=device)
        sigma, dsigma = noise(t)
        
        print(f"Time t: {t}")
        print(f"Sigma: {sigma}")
        print(f"DSigma: {dsigma}")
        
        # Corrupt data
        xt = graph.sample_transition(x0, sigma[:, None])
        print(f"Corrupted data xt: shape {xt.shape}, range [{torch.min(xt)}, {torch.max(xt)}]")
        
        # Forward pass
        model_output = model(xt, sigma, use_subs=True)
        print(f"Model output: shape {model_output.shape}, range [{torch.min(model_output):.4f}, {torch.max(model_output):.4f}]")
        
        # Check for NaN in model output
        if torch.any(torch.isnan(model_output)):
            print(f"🚨 NaN detected in model output")
            return False
        
        # Compute loss
        loss, curric_dict = subs_loss_with_curriculum(
            model_output=model_output,
            x0=x0,
            sigma=sigma,
            noise_schedule=noise,
            training_step=100
        )
        
        print(f"Loss: {loss.item():.6f}")
        
        if torch.isnan(loss):
            print(f"🚨 NaN detected in loss")
            return False
        
        print("✅ Training step simulation passed")
        return True
        
    except Exception as e:
        print(f"❌ Training step simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all NaN debugging tests."""
    print("🔍 Running NaN debugging tests...\n")
    
    tests = [
        test_noise_schedule,
        test_graph_operations,
        test_model_initialization,
        test_training_step_simulation,
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
        print("🎉 All tests passed! The model should not produce NaN values.")
        print("\n💡 If you're still seeing NaN values during training, the issue might be:")
        print("   1. Learning rate too high causing gradient explosion")
        print("   2. Data preprocessing issues")
        print("   3. Mixed precision training issues")
        print("   4. Hardware-specific numerical precision issues")
        return 0
    else:
        print("💥 Some tests failed! This indicates where NaN values are coming from.")
        return 1

if __name__ == "__main__":
    exit(main())
