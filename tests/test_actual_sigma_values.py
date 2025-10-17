#!/usr/bin/env python3
"""
Test what sigma values you're actually getting during training.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

def test_actual_training_sigmas():
    """Test the actual sigma values you get during training."""
    print("🧪 Testing Actual Training Sigma Values")
    print("=" * 50)
    
    try:
        from protlig_ddiff.processing import noise_lib
        from protlig_ddiff.utils.config_utils import load_config
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False
    
    # Load your actual config
    try:
        config = load_config('configs/config_protein.yaml')
        print(f"✅ Loaded config: noise type = {config.noise.type}")
    except Exception as e:
        print(f"⚠️  Could not load config, using defaults: {e}")
        # Create mock config
        class MockConfig:
            def __init__(self):
                self.noise = type('obj', (object,), {
                    'type': 'cosine',
                    'sigma_min': 1e-4,
                    'sigma_max': 2.0,
                    'eps': 1e-3
                })()
        config = MockConfig()
    
    # Create the same noise schedule as your training
    noise_type = config.noise.type.lower()
    if noise_type == 'loglinear':
        noise = noise_lib.LogLinearNoise(eps=getattr(config.noise, 'eps', 1e-3))
        print("🔧 Using LogLinear noise")
    elif noise_type == 'cosine':
        sigma_min = getattr(config.noise, 'sigma_min', 1e-4)
        sigma_max = getattr(config.noise, 'sigma_max', 2.0)
        noise = noise_lib.CosineNoise(sigma_min=sigma_min, sigma_max=sigma_max)
        print(f"🔧 Using Cosine noise: σ_min={sigma_min}, σ_max={sigma_max}")
    elif noise_type == 'geometric':
        sigma_min = getattr(config.noise, 'sigma_min', 1e-3)
        sigma_max = getattr(config.noise, 'sigma_max', 1.0)
        noise = noise_lib.GeometricNoise(sigma_min=sigma_min, sigma_max=sigma_max)
        print(f"🔧 Using Geometric noise: σ_min={sigma_min}, σ_max={sigma_max}")
    else:
        print(f"⚠️  Unknown noise type: {noise_type}, using LogLinear")
        noise = noise_lib.LogLinearNoise()
    
    print()
    
    # Simulate training batches
    batch_size = 32
    num_batches = 10
    device = torch.device('cpu')
    
    print(f"🔍 Simulating {num_batches} training batches (batch_size={batch_size}):")
    print()
    
    all_sigmas = []
    all_t_values = []
    
    for batch_idx in range(num_batches):
        # Simulate the same timestep sampling as your training
        # Using rank_based mode (current default)
        seed = 42  # Your config seed
        current_step = 1000 + batch_idx * 10  # Simulate training steps
        rank = 0  # Main rank
        world_size = 1  # Single GPU simulation
        
        generator = torch.Generator(device=device).manual_seed(
            seed + current_step * world_size + rank
        )
        t = torch.rand(batch_size, device=device, generator=generator)
        sigma, dsigma = noise(t)
        
        # Collect statistics
        all_sigmas.extend(sigma.tolist())
        all_t_values.extend(t.tolist())
        
        # Print batch statistics
        sigma_min = sigma.min().item()
        sigma_max = sigma.max().item()
        sigma_mean = sigma.mean().item()
        high_sigma_count = (sigma > 0.9).sum().item()
        very_high_sigma_count = (sigma > 0.95).sum().item()
        
        print(f"   Batch {batch_idx:2d}: σ ∈ [{sigma_min:.4f}, {sigma_max:.4f}], mean={sigma_mean:.4f}")
        print(f"            >0.9: {high_sigma_count:2d}/{batch_size} ({100*high_sigma_count/batch_size:4.1f}%), >0.95: {very_high_sigma_count:2d}/{batch_size} ({100*very_high_sigma_count/batch_size:4.1f}%)")
        
        # Show some specific values
        if batch_idx < 3:  # Only for first few batches
            print(f"            First 5 σ: {[f'{s:.3f}' for s in sigma[:5].tolist()]}")
            print(f"            First 5 t: {[f'{t:.3f}' for t in t[:5].tolist()]}")
        print()
    
    # Overall statistics
    all_sigmas = np.array(all_sigmas)
    all_t_values = np.array(all_t_values)
    
    print("📊 Overall Statistics:")
    print(f"   Total samples: {len(all_sigmas)}")
    print(f"   Sigma range: [{all_sigmas.min():.6f}, {all_sigmas.max():.6f}]")
    print(f"   Sigma mean: {all_sigmas.mean():.6f}")
    print(f"   Sigma std: {all_sigmas.std():.6f}")
    print()
    
    # Count high sigma values
    high_sigma_count = np.sum(all_sigmas > 0.9)
    very_high_sigma_count = np.sum(all_sigmas > 0.95)
    extreme_high_sigma_count = np.sum(all_sigmas > 0.99)
    
    print(f"   Sigmas > 0.9: {high_sigma_count}/{len(all_sigmas)} ({100*high_sigma_count/len(all_sigmas):.1f}%)")
    print(f"   Sigmas > 0.95: {very_high_sigma_count}/{len(all_sigmas)} ({100*very_high_sigma_count/len(all_sigmas):.1f}%)")
    print(f"   Sigmas > 0.99: {extreme_high_sigma_count}/{len(all_sigmas)} ({100*extreme_high_sigma_count/len(all_sigmas):.1f}%)")
    print()
    
    # Show distribution
    print("📈 Sigma Distribution:")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, float('inf')]
    bin_labels = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', 
                  '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-0.95', '0.95-1.0', '>1.0']
    
    for i in range(len(bins)-1):
        count = np.sum((all_sigmas >= bins[i]) & (all_sigmas < bins[i+1]))
        percentage = 100 * count / len(all_sigmas)
        bar = '█' * int(percentage / 2)  # Scale bar
        print(f"   {bin_labels[i]:>8}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    print()
    
    # Conclusion
    if high_sigma_count > 0:
        print("✅ **You ARE getting sigmas near 0.9!**")
        print(f"   In fact, {100*high_sigma_count/len(all_sigmas):.1f}% of your samples have σ > 0.9")
        print()
        print("🤔 If you think you're not seeing them, it might be because:")
        print("   1. You're not logging sigma values during training")
        print("   2. You're only looking at a few samples")
        print("   3. Your training logs are not showing sigma statistics")
        print()
        print("💡 Try adding this to your training loop:")
        print("   if step % 100 == 0:")
        print("       high_count = (sigma > 0.9).sum()")
        print("       print(f'Step {step}: {high_count}/{batch_size} sigmas > 0.9')")
    else:
        print("❌ **You're NOT getting sigmas near 0.9**")
        print("   This suggests there might be an issue with:")
        print("   1. Your noise schedule configuration")
        print("   2. Timestep sampling implementation")
        print("   3. Curriculum learning (if enabled)")
        print()
        print("🔧 Try switching to geometric noise:")
        print("   noise:")
        print("     type: 'geometric'")
        print("     sigma_min: 1e-4")
        print("     sigma_max: 1.0")
    
    return True

def main():
    """Run the test."""
    try:
        test_actual_training_sigmas()
        print("\n🎉 Test completed!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
