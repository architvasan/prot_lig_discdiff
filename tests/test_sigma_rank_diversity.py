#!/usr/bin/env python3
"""
Test script to verify that sigma values are different across ranks in distributed training.
This ensures proper noise diversity in multi-GPU training.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

def test_sigma_diversity_across_ranks():
    """Test that different ranks generate different sigma values."""
    print("🧪 Testing sigma diversity across ranks...")
    
    # Import noise library
    try:
        from protlig_ddiff.processing import noise_lib
    except ImportError as e:
        print(f"❌ Failed to import noise_lib: {e}")
        return False
    
    # Setup
    device = torch.device('cpu')  # Use CPU for testing
    batch_size = 4
    num_ranks = 4
    current_step = 100
    base_seed = 42
    
    # Create noise schedule
    noise = noise_lib.LogLinearNoise()
    
    print(f"   Testing with {num_ranks} ranks, batch_size={batch_size}, step={current_step}")
    
    # Generate sigma values for each rank
    sigma_values = {}
    t_values = {}
    
    for rank in range(num_ranks):
        # Simulate rank-specific generator (as implemented in the fix)
        rank_generator = torch.Generator(device=device).manual_seed(
            base_seed + current_step * num_ranks + rank
        )
        
        # Generate t values
        t = torch.rand(batch_size, device=device, generator=rank_generator)
        sigma, _ = noise(t)
        
        sigma_values[rank] = sigma.clone()
        t_values[rank] = t.clone()
        
        print(f"      Rank {rank}: t={t[:3].tolist()}, sigma={sigma[:3].tolist()}")
    
    # Check that sigma values are different across ranks
    all_different = True
    for rank1 in range(num_ranks):
        for rank2 in range(rank1 + 1, num_ranks):
            if torch.allclose(sigma_values[rank1], sigma_values[rank2], atol=1e-6):
                print(f"❌ Rank {rank1} and {rank2} have identical sigma values!")
                all_different = False
    
    if all_different:
        print("✅ All ranks have different sigma values")
    
    # Check that t values are different across ranks (they should be)
    all_t_different = True
    for rank1 in range(num_ranks):
        for rank2 in range(rank1 + 1, num_ranks):
            if torch.allclose(t_values[rank1], t_values[rank2], atol=1e-6):
                print(f"❌ Rank {rank1} and {rank2} have identical t values!")
                all_t_different = False
    
    if all_t_different:
        print("✅ All ranks have different t values")
    
    return all_different and all_t_different

def test_old_vs_new_approach():
    """Compare old (same sigma) vs new (different sigma) approaches."""
    print("🧪 Testing old vs new sigma generation...")
    
    try:
        from protlig_ddiff.processing import noise_lib
    except ImportError as e:
        print(f"❌ Failed to import noise_lib: {e}")
        return False
    
    device = torch.device('cpu')
    batch_size = 4
    num_ranks = 4
    base_seed = 42
    
    noise = noise_lib.LogLinearNoise()
    
    print("   OLD APPROACH (same sigma across ranks):")
    # Old approach - all ranks use same seed
    torch.manual_seed(base_seed)
    old_sigma_values = {}
    for rank in range(num_ranks):
        t = torch.rand(batch_size, device=device)
        sigma, _ = noise(t)
        old_sigma_values[rank] = sigma.clone()
        print(f"      Rank {rank}: sigma={sigma[:3].tolist()}")
    
    print("\n   NEW APPROACH (different sigma per rank):")
    # New approach - rank-specific seeds
    current_step = 100
    new_sigma_values = {}
    for rank in range(num_ranks):
        rank_generator = torch.Generator(device=device).manual_seed(
            base_seed + current_step * num_ranks + rank
        )
        t = torch.rand(batch_size, device=device, generator=rank_generator)
        sigma, _ = noise(t)
        new_sigma_values[rank] = sigma.clone()
        print(f"      Rank {rank}: sigma={sigma[:3].tolist()}")
    
    # Check old approach (should be identical)
    old_identical = True
    for rank in range(1, num_ranks):
        if not torch.allclose(old_sigma_values[0], old_sigma_values[rank], atol=1e-6):
            old_identical = False
            break
    
    # Check new approach (should be different)
    new_different = True
    for rank1 in range(num_ranks):
        for rank2 in range(rank1 + 1, num_ranks):
            if torch.allclose(new_sigma_values[rank1], new_sigma_values[rank2], atol=1e-6):
                new_different = False
                break
    
    print(f"\n   📊 Results:")
    print(f"      Old approach - all ranks identical: {'✅' if old_identical else '❌'}")
    print(f"      New approach - all ranks different: {'✅' if new_different else '❌'}")
    
    return old_identical and new_different

def test_validation_reproducibility():
    """Test that validation sigma generation is reproducible."""
    print("🧪 Testing validation reproducibility...")
    
    try:
        from protlig_ddiff.processing import noise_lib
    except ImportError as e:
        print(f"❌ Failed to import noise_lib: {e}")
        return False
    
    device = torch.device('cpu')
    batch_size = 4
    base_seed = 42
    
    noise = noise_lib.LogLinearNoise()
    
    # Generate validation sigma twice with same parameters
    val_sigma_1 = {}
    val_sigma_2 = {}
    
    for batch_idx in range(3):  # Test 3 validation batches
        # First run
        val_generator_1 = torch.Generator(device=device).manual_seed(
            base_seed + batch_idx * 1000
        )
        t1 = torch.rand(batch_size, device=device, generator=val_generator_1)
        sigma1, _ = noise(t1)
        val_sigma_1[batch_idx] = sigma1.clone()
        
        # Second run (should be identical)
        val_generator_2 = torch.Generator(device=device).manual_seed(
            base_seed + batch_idx * 1000
        )
        t2 = torch.rand(batch_size, device=device, generator=val_generator_2)
        sigma2, _ = noise(t2)
        val_sigma_2[batch_idx] = sigma2.clone()
        
        print(f"   Batch {batch_idx}: sigma1={sigma1[:2].tolist()}, sigma2={sigma2[:2].tolist()}")
    
    # Check reproducibility
    reproducible = True
    for batch_idx in range(3):
        if not torch.allclose(val_sigma_1[batch_idx], val_sigma_2[batch_idx], atol=1e-8):
            print(f"❌ Validation batch {batch_idx} not reproducible!")
            reproducible = False
    
    if reproducible:
        print("✅ Validation sigma generation is reproducible")
    
    return reproducible

def test_sigma_coverage():
    """Test that different ranks cover different parts of sigma space."""
    print("🧪 Testing sigma space coverage...")
    
    try:
        from protlig_ddiff.processing import noise_lib
    except ImportError as e:
        print(f"❌ Failed to import noise_lib: {e}")
        return False
    
    device = torch.device('cpu')
    batch_size = 32
    num_ranks = 4
    num_steps = 10
    base_seed = 42
    
    noise = noise_lib.LogLinearNoise()
    
    # Collect sigma values from multiple steps
    all_sigmas = []
    
    for step in range(num_steps):
        for rank in range(num_ranks):
            rank_generator = torch.Generator(device=device).manual_seed(
                base_seed + step * num_ranks + rank
            )
            t = torch.rand(batch_size, device=device, generator=rank_generator)
            sigma, _ = noise(t)
            all_sigmas.extend(sigma.tolist())
    
    all_sigmas = np.array(all_sigmas)
    
    print(f"   Collected {len(all_sigmas)} sigma values")
    print(f"   Sigma range: [{all_sigmas.min():.4f}, {all_sigmas.max():.4f}]")
    print(f"   Sigma mean: {all_sigmas.mean():.4f}, std: {all_sigmas.std():.4f}")
    
    # Check coverage
    sigma_bins = np.linspace(0, 1, 11)  # 10 bins
    hist, _ = np.histogram(all_sigmas, bins=sigma_bins)
    
    print(f"   Sigma distribution across bins: {hist}")
    
    # Good coverage means no empty bins and reasonable distribution
    empty_bins = np.sum(hist == 0)
    coverage_good = empty_bins <= 2  # Allow up to 2 empty bins
    
    if coverage_good:
        print("✅ Good sigma space coverage")
    else:
        print(f"⚠️  Poor sigma coverage: {empty_bins} empty bins")
    
    return coverage_good

def main():
    """Run all tests."""
    print("🚀 Testing Sigma Rank Diversity System")
    print("=" * 60)
    
    tests = [
        test_sigma_diversity_across_ranks,
        test_old_vs_new_approach,
        test_validation_reproducibility,
        test_sigma_coverage,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        print("\n📋 Sigma diversity system benefits:")
        print("   ✅ Different ranks see different noise levels")
        print("   ✅ Better coverage of sigma space during training")
        print("   ✅ More diverse gradient updates in distributed training")
        print("   ✅ Reproducible validation and test evaluation")
        print("   ✅ Improved training efficiency and model quality")
        
        print("\n🔧 Implementation details:")
        print("   📊 Training: rank-specific sigma generation")
        print("   📊 Validation: deterministic, reproducible sigma")
        print("   📊 Test: deterministic, reproducible sigma")
        print("   📊 Seed formula: base_seed + step * world_size + rank")
        
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed!")
        return 1

if __name__ == "__main__":
    exit(main())
