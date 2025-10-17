#!/usr/bin/env python3
"""
Debug script to investigate sigma range issues.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

def test_noise_schedules():
    """Test different noise schedules to see sigma ranges."""
    print("🧪 Testing Noise Schedule Sigma Ranges")
    print("=" * 50)
    
    try:
        from protlig_ddiff.processing import noise_lib
    except ImportError as e:
        print(f"❌ Failed to import noise_lib: {e}")
        return False
    
    # Test different noise schedules
    noise_schedules = {
        'LogLinear (default)': noise_lib.LogLinearNoise(eps=1e-3),
        'Cosine (sigma_min=1e-4, sigma_max=2.0)': noise_lib.CosineNoise(sigma_min=1e-4, sigma_max=2.0),
        'Cosine (sigma_min=1e-4, sigma_max=1.0)': noise_lib.CosineNoise(sigma_min=1e-4, sigma_max=1.0),
        'Geometric (sigma_min=1e-3, sigma_max=1.0)': noise_lib.GeometricNoise(sigma_min=1e-3, sigma_max=1.0),
    }
    
    # Test timesteps from 0 to 1
    t_values = torch.linspace(0.001, 0.999, 100)
    
    print("📊 Testing sigma ranges for different noise schedules:")
    print()
    
    for name, noise in noise_schedules.items():
        print(f"🔍 {name}:")
        
        try:
            sigmas = []
            dsigmas = []
            
            for t in t_values:
                sigma, dsigma = noise(t.unsqueeze(0))
                sigmas.append(sigma.item())
                dsigmas.append(dsigma.item())
            
            sigmas = np.array(sigmas)
            dsigmas = np.array(dsigmas)
            
            print(f"   📊 Sigma range: [{sigmas.min():.6f}, {sigmas.max():.6f}]")
            print(f"   📊 Mean sigma: {sigmas.mean():.6f}")
            print(f"   📊 Std sigma: {sigmas.std():.6f}")
            
            # Check for high sigma values
            high_sigma_count = np.sum(sigmas > 0.9)
            very_high_sigma_count = np.sum(sigmas > 0.95)
            
            print(f"   📊 Sigmas > 0.9: {high_sigma_count}/100 ({high_sigma_count}%)")
            print(f"   📊 Sigmas > 0.95: {very_high_sigma_count}/100 ({very_high_sigma_count}%)")
            
            # Show some specific values
            print(f"   📊 t=0.1 → σ={sigmas[9]:.6f}")
            print(f"   📊 t=0.5 → σ={sigmas[49]:.6f}")
            print(f"   📊 t=0.9 → σ={sigmas[89]:.6f}")
            print(f"   📊 t=0.99 → σ={sigmas[98]:.6f}")
            
        except Exception as e:
            print(f"   ❌ Error testing {name}: {e}")
        
        print()
    
    return True

def test_loglinear_issue():
    """Test the specific LogLinear noise issue."""
    print("🔍 Investigating LogLinear Noise Issue")
    print("=" * 40)
    
    try:
        from protlig_ddiff.processing import noise_lib
    except ImportError as e:
        print(f"❌ Failed to import noise_lib: {e}")
        return False
    
    noise = noise_lib.LogLinearNoise(eps=1e-3)
    
    print("📊 LogLinear noise formula analysis:")
    print("   total_noise(t) = -log(1 - (1 - eps) * t)")
    print("   where eps = 1e-3")
    print()
    
    # Test edge cases
    test_values = [0.001, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999]
    
    for t in test_values:
        try:
            t_tensor = torch.tensor([t])
            sigma, dsigma = noise(t_tensor)
            
            # Manual calculation
            eps = 1e-3
            manual_sigma = -torch.log1p(-(1 - eps) * t)
            
            print(f"   t={t:6.4f} → σ={sigma.item():8.6f} (manual: {manual_sigma.item():8.6f})")
            
            # Check if we're approaching the asymptote
            if t > 0.999:
                inner_value = 1 - (1 - eps) * t
                print(f"      Inner value (1 - (1-eps)*t): {inner_value:.10f}")
                if inner_value <= 0:
                    print(f"      ⚠️  Inner value ≤ 0, log will be undefined!")
            
        except Exception as e:
            print(f"   t={t:6.4f} → ERROR: {e}")
    
    print()
    print("🔍 The issue: LogLinear noise has an asymptote!")
    print("   As t approaches 1/(1-eps) ≈ 1.001, sigma approaches infinity")
    print("   But t is clamped to [0, 1], so we never reach very high sigmas")
    print()
    
    # Show the theoretical maximum
    eps = 1e-3
    max_t_theoretical = 1 / (1 - eps)
    max_t_practical = 0.999
    
    max_sigma_theoretical = float('inf')
    max_sigma_practical = -np.log1p(-(1 - eps) * max_t_practical)
    
    print(f"   📊 Theoretical max t: {max_t_theoretical:.6f}")
    print(f"   📊 Practical max t: {max_t_practical:.6f}")
    print(f"   📊 Max sigma at t=0.999: {max_sigma_practical:.6f}")
    
    return True

def test_cosine_issue():
    """Test the Cosine noise implementation."""
    print("🔍 Investigating Cosine Noise Implementation")
    print("=" * 40)
    
    try:
        from protlig_ddiff.processing import noise_lib
    except ImportError as e:
        print(f"❌ Failed to import noise_lib: {e}")
        return False
    
    # Test with your config values
    noise = noise_lib.CosineNoise(sigma_min=1e-4, sigma_max=2.0)
    
    print("📊 Cosine noise formula analysis:")
    print("   σ(t) = σ_max + 0.5(σ_min - σ_max)[1 + cos(π t/T)]")
    print("   where σ_min=1e-4, σ_max=2.0, T=1.0")
    print()
    
    # Test key points
    test_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for t in test_values:
        try:
            t_tensor = torch.tensor([t])
            sigma, dsigma = noise(t_tensor)
            
            # Manual calculation
            import math
            sigma_min, sigma_max = 1e-4, 2.0
            T = 1.0
            manual_sigma = sigma_max + 0.5 * (sigma_min - sigma_max) * (1 + math.cos(math.pi * t / T))
            
            print(f"   t={t:4.2f} → σ={sigma.item():8.6f} (manual: {manual_sigma:8.6f})")
            
        except Exception as e:
            print(f"   t={t:4.2f} → ERROR: {e}")
    
    print()
    print("🔍 Cosine noise behavior:")
    print("   t=0.0 → σ = σ_max + 0.5(σ_min - σ_max)(1 + 1) = σ_min")
    print("   t=0.5 → σ = σ_max + 0.5(σ_min - σ_max)(1 + 0) = (σ_min + σ_max)/2")
    print("   t=1.0 → σ = σ_max + 0.5(σ_min - σ_max)(1 - 1) = σ_max")
    print()
    print("   So cosine goes: σ_min → (σ_min + σ_max)/2 → σ_max")
    print("   With your values: 1e-4 → 1.00005 → 2.0")
    
    return True

def recommend_fixes():
    """Recommend fixes for the sigma range issue."""
    print("🔧 Recommended Fixes")
    print("=" * 30)
    
    print("📌 **Issue**: You're not seeing sigmas near 0.9 because:")
    print("   1. LogLinear noise has mathematical limitations")
    print("   2. Cosine noise might have implementation issues")
    print()
    
    print("🎯 **Solution 1: Fix LogLinear Noise** (Recommended)")
    print("   Problem: LogLinear asymptote prevents high sigmas")
    print("   Fix: Use a modified LogLinear or switch to Geometric")
    print()
    print("   # Change your config to:")
    print("   noise:")
    print("     type: 'geometric'")
    print("     sigma_min: 1e-4")
    print("     sigma_max: 1.0")
    print()
    
    print("🎯 **Solution 2: Fix Cosine Implementation**")
    print("   Problem: Cosine implementation might be backwards")
    print("   Fix: Verify the cosine formula is correct")
    print()
    
    print("🎯 **Solution 3: Use Custom Noise Schedule**")
    print("   Create a simple linear interpolation:")
    print("   σ(t) = σ_min + t * (σ_max - σ_min)")
    print("   This guarantees full range coverage")
    print()
    
    print("🧪 **Test Your Fix:**")
    print("   1. Change noise type in config")
    print("   2. Run: python debug_sigma_range.py")
    print("   3. Look for 'Sigmas > 0.9' percentage")
    print("   4. Should see ~10% of samples > 0.9")

def main():
    """Run all tests."""
    print("🚀 Debugging Sigma Range Issues")
    print("=" * 60)
    
    tests = [
        test_noise_schedules,
        test_loglinear_issue,
        test_cosine_issue,
        recommend_fixes,
    ]
    
    for test in tests:
        try:
            test()
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("🎉 Debugging completed!")
    print()
    print("💡 **Quick Fix**: Try changing your config to:")
    print("   noise:")
    print("     type: 'geometric'")
    print("     sigma_min: 1e-4")
    print("     sigma_max: 1.0")
    print()
    print("   This should give you the full sigma range including values near 0.9!")

if __name__ == "__main__":
    main()
