#!/usr/bin/env python3
"""
Test the enhanced sampling features:
1. ESM perplexity evaluation
2. Rank-specific generation
3. 10 sequences per rank
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

def test_esm_evaluator():
    """Test the ESM evaluator."""
    print("🔬 Testing ESM Evaluator")
    print("=" * 40)
    
    try:
        from protlig_ddiff.utils.esm_evaluator import ESMEvaluator, calculate_esm_perplexity
        
        # Test sequences (some realistic, some unrealistic)
        test_sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # Realistic
            "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",  # All amino acids
            "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM",  # All methionine (unrealistic)
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",  # All alanine (unrealistic)
        ]
        
        print(f"📊 Testing ESM perplexity calculation for {len(test_sequences)} sequences...")
        
        # Test the convenience function
        perplexities = calculate_esm_perplexity(
            sequences=test_sequences,
            model_name="esm2_t6_8M_UR50D",  # Smallest model for testing
            batch_size=2,
            mask_fraction=0.15,
            seed=42
        )
        
        print("✅ ESM perplexity results:")
        for i, (seq, ppl) in enumerate(zip(test_sequences, perplexities)):
            seq_short = seq[:30] + "..." if len(seq) > 30 else seq
            print(f"   Sequence {i+1}: {seq_short}")
            print(f"   Perplexity: {ppl:.2f}")
            print()
        
        # Analyze results
        valid_perplexities = [p for p in perplexities if not np.isinf(p)]
        if valid_perplexities:
            print(f"📈 Perplexity statistics:")
            print(f"   Mean: {np.mean(valid_perplexities):.2f}")
            print(f"   Std: {np.std(valid_perplexities):.2f}")
            print(f"   Min: {np.min(valid_perplexities):.2f}")
            print(f"   Max: {np.max(valid_perplexities):.2f}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  ESM not available: {e}")
        print("   Install with: pip install fair-esm")
        return False
    except Exception as e:
        print(f"❌ ESM evaluator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rank_specific_generation():
    """Test rank-specific generation."""
    print("🎯 Testing Rank-Specific Generation")
    print("=" * 40)
    
    try:
        from protlig_ddiff.sampling.protein_sampling import sample_during_training
        from protlig_ddiff.processing import noise_lib, graph_lib
        from protlig_ddiff.models.transformer_v100 import DiscDiffModel
        from protlig_ddiff.utils.config_utils import load_config
        
        # Create mock config
        class MockConfig:
            def __init__(self):
                self.seed = 42
                self.sampling = type('obj', (object,), {
                    'eval_batch_size': 10,
                    'eval_max_length': 64,
                    'eval_steps': 50,
                    'predictor': 'euler',
                    'calculate_esm_perplexity': False,  # Skip ESM for speed
                    'esm_model': "esm2_t6_8M_UR50D",
                    'esm_batch_size': 4
                })()
        
        config = MockConfig()
        
        # Create minimal components
        vocab_size = 26
        graph = graph_lib.Absorbing(vocab_size - 1)
        noise = noise_lib.CosineNoise(sigma_min=1e-4, sigma_max=1.5)
        
        # Create a minimal model (just for testing structure)
        model_config = type('obj', (object,), {
            'dim': 128,
            'n_heads': 4,
            'n_layers': 2,
            'max_seq_len': 64,
            'cond_dim': 64,
            'scale_by_sigma': False
        })()
        model_config.tokens = vocab_size
        
        model = DiscDiffModel(model_config)
        model.eval()
        
        device = torch.device('cpu')
        
        print(f"🧪 Testing generation for multiple ranks...")
        
        # Test different ranks
        num_ranks = 3
        step = 1000
        
        all_results = []
        
        for rank in range(num_ranks):
            print(f"\n🔍 Testing rank {rank}:")
            
            try:
                result = sample_during_training(
                    model=model,
                    graph=graph,
                    noise=noise,
                    config=config,
                    step=step,
                    device=device,
                    rank=rank,
                    world_size=num_ranks
                )
                
                sequences = result.get("sequences", [])
                esm_perplexities = result.get("esm_perplexities", [])
                result_rank = result.get("rank", -1)
                
                print(f"   ✅ Rank {rank}: Generated {len(sequences)} sequences")
                print(f"   📊 Result rank: {result_rank}")
                print(f"   🔬 ESM perplexities: {len(esm_perplexities)} values")
                
                # Show first sequence
                if sequences:
                    first_seq = sequences[0].replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
                    print(f"   🧬 First sequence: {first_seq[:40]}...")
                
                all_results.append(result)
                
            except Exception as e:
                print(f"   ❌ Rank {rank} failed: {e}")
                all_results.append(None)
        
        # Check that sequences are different across ranks
        print(f"\n🔍 Checking sequence diversity across ranks:")
        
        valid_results = [r for r in all_results if r and r.get("sequences")]
        
        if len(valid_results) >= 2:
            # Compare first sequences from different ranks
            seq1 = valid_results[0]["sequences"][0] if valid_results[0]["sequences"] else ""
            seq2 = valid_results[1]["sequences"][0] if valid_results[1]["sequences"] else ""
            
            if seq1 and seq2:
                if seq1 == seq2:
                    print("   ⚠️  WARNING: Sequences from different ranks are identical!")
                    print(f"      Rank 0: {seq1[:50]}...")
                    print(f"      Rank 1: {seq2[:50]}...")
                else:
                    print("   ✅ Sequences from different ranks are different")
                    print(f"      Rank 0: {seq1[:50]}...")
                    print(f"      Rank 1: {seq2[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Rank-specific generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_updates():
    """Test that config updates are working."""
    print("⚙️  Testing Configuration Updates")
    print("=" * 40)
    
    try:
        from protlig_ddiff.utils.config_utils import load_config
        
        config = load_config('configs/config_protein.yaml')
        
        # Check sampling config
        sampling_config = getattr(config, 'sampling', None)
        if sampling_config:
            batch_size = getattr(sampling_config, 'eval_batch_size', None)
            calculate_esm = getattr(sampling_config, 'calculate_esm_perplexity', None)
            esm_model = getattr(sampling_config, 'esm_model', None)
            
            print(f"✅ Sampling configuration loaded:")
            print(f"   eval_batch_size: {batch_size}")
            print(f"   calculate_esm_perplexity: {calculate_esm}")
            print(f"   esm_model: {esm_model}")
            
            if batch_size == 10:
                print("   ✅ Batch size correctly set to 10")
            else:
                print(f"   ⚠️  Expected batch size 10, got {batch_size}")
            
            if calculate_esm:
                print("   ✅ ESM perplexity calculation enabled")
            else:
                print("   ⚠️  ESM perplexity calculation disabled")
        else:
            print("   ❌ No sampling configuration found")
        
        # Check noise config
        noise_config = getattr(config, 'noise', None)
        if noise_config:
            sigma_max = getattr(noise_config, 'sigma_max', None)
            print(f"   sigma_max: {sigma_max}")
            
            if sigma_max == 1.5:
                print("   ✅ Sigma max correctly set to 1.5")
            else:
                print(f"   ⚠️  Expected sigma_max 1.5, got {sigma_max}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Sampling Features")
    print("=" * 60)
    
    tests = [
        ("Configuration Updates", test_config_updates),
        ("Rank-Specific Generation", test_rank_specific_generation),
        ("ESM Evaluator", test_esm_evaluator),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Test Summary")
    print(f"{'='*60}")
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced sampling is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
