#!/usr/bin/env python3
"""
Test script to compare SUBS vs Score-based sampling.

This script demonstrates the difference between:
1. Original sampling: use_subs=False (score-based parameterization)
2. Fixed sampling: use_subs=True (SUBS parameterization, consistent with training)
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

def create_mock_model():
    """Create a mock model that demonstrates the parameterization difference."""
    
    class MockModel:
        def __init__(self, vocab_size=26):
            self.vocab_size = vocab_size
            self.mask_index = vocab_size - 1
            
        def __call__(self, x, sigma, use_subs=False):
            batch_size, seq_len = x.shape
            
            # Create realistic logits (some positions favor certain amino acids)
            logits = torch.randn(batch_size, seq_len, self.vocab_size) * 2.0
            
            # Make some amino acids more likely (realistic protein bias)
            # Favor common amino acids: A(0), L(11), V(21), I(8), G(6)
            common_aa = [0, 11, 21, 8, 6]
            for aa in common_aa:
                logits[:, :, aa] += 1.0
            
            if use_subs:
                # SUBS parameterization
                return self._subs_parameterization(logits, x)
            else:
                # Score-based parameterization
                return self._score_parameterization(logits, x)
        
        def _subs_parameterization(self, logits, xt):
            """SUBS parameterization for MDLM loss"""
            neg_infinity = -1000000.0
            
            # Step 1: Set log prob at mask index to -infinity
            logits[:, :, self.mask_index] = neg_infinity
            
            # Step 2: Normalize logits to log probabilities
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            
            # Step 3: For unmasked positions, set all to -inf except original token
            unmasked_mask = (xt != self.mask_index)
            logits[unmasked_mask] = neg_infinity
            
            # Set original token log prob to 0 (prob = 1) for unmasked positions
            batch_indices = torch.arange(xt.shape[0]).unsqueeze(1).expand(-1, xt.shape[1])
            seq_indices = torch.arange(xt.shape[1]).unsqueeze(0).expand(xt.shape[0], -1)
            logits[batch_indices[unmasked_mask], seq_indices[unmasked_mask], xt[unmasked_mask]] = 0.0
            
            return logits
        
        def _score_parameterization(self, logits, xt):
            """Score-based parameterization with absorbing state logic"""
            # Zero out positions that match input (absorbing state logic)
            indices_expanded = xt.unsqueeze(-1)  # [batch, seq, 1]
            zeros = torch.zeros_like(logits[..., :1])  # [batch, seq, 1]
            logits = torch.scatter(logits, -1, indices_expanded, zeros)
            return logits
    
    return MockModel()

def test_sampling_comparison():
    """Compare SUBS vs Score-based sampling."""
    
    print("🧪 Testing SUBS vs Score-based Sampling")
    print("=" * 60)
    
    # Create mock components
    model = create_mock_model()
    vocab_size = 26
    mask_index = vocab_size - 1
    
    # Create test input (some positions masked, some not)
    batch_size, seq_len = 2, 8
    x = torch.randint(0, vocab_size-1, (batch_size, seq_len))
    x[0, :3] = mask_index  # First 3 positions masked
    x[1, 4:6] = mask_index  # Positions 4-5 masked
    
    sigma = torch.tensor([0.5, 0.8])  # Different noise levels
    
    print(f"📊 Test Setup:")
    print(f"   Input tokens: {x}")
    print(f"   Mask index: {mask_index}")
    print(f"   Masked positions: {(x == mask_index).sum().item()}/{x.numel()}")
    print(f"   Sigma values: {sigma}")
    
    # Test both parameterizations
    print(f"\n🎯 SUBS Parameterization (use_subs=True):")
    print("-" * 40)
    
    with torch.no_grad():
        subs_output = model(x, sigma, use_subs=True)
        subs_probs = torch.exp(subs_output)  # Convert log probs to probs
        
        print(f"   Output shape: {subs_output.shape}")
        print(f"   Log prob range: [{subs_output[subs_output > -999999].min().item():.3f}, {subs_output[subs_output > -999999].max().item():.3f}]")
        print(f"   Prob range: [{subs_probs.min().item():.6f}, {subs_probs.max().item():.6f}]")
        print(f"   Prob sums: {subs_probs.sum(dim=-1).mean().item():.6f} (should be ~1)")
        
        # Check masked position distributions
        masked_pos = (x == mask_index)
        if masked_pos.any():
            first_masked = torch.where(masked_pos)[0][0], torch.where(masked_pos)[1][0]
            masked_dist = subs_probs[first_masked]
            entropy = -torch.sum(masked_dist * torch.log(masked_dist + 1e-8)).item()
            top3 = torch.topk(masked_dist, 3)
            print(f"   Masked pos [{first_masked[0]}, {first_masked[1]}] entropy: {entropy:.3f}")
            print(f"   Top-3 tokens: {top3.indices.tolist()} (probs: {[f'{p:.3f}' for p in top3.values.tolist()]})")
    
    print(f"\n⚡ Score-based Parameterization (use_subs=False):")
    print("-" * 40)
    
    with torch.no_grad():
        score_output = model(x, sigma, use_subs=False)
        score_probs = F.softmax(score_output, dim=-1)  # Convert scores to probs
        
        print(f"   Output shape: {score_output.shape}")
        print(f"   Score range: [{score_output.min().item():.3f}, {score_output.max().item():.3f}]")
        print(f"   Prob range: [{score_probs.min().item():.6f}, {score_probs.max().item():.6f}]")
        print(f"   Prob sums: {score_probs.sum(dim=-1).mean().item():.6f} (should be ~1)")
        
        # Check masked position distributions
        if masked_pos.any():
            masked_dist = score_probs[first_masked]
            entropy = -torch.sum(masked_dist * torch.log(masked_dist + 1e-8)).item()
            top3 = torch.topk(masked_dist, 3)
            print(f"   Masked pos [{first_masked[0]}, {first_masked[1]}] entropy: {entropy:.3f}")
            print(f"   Top-3 tokens: {top3.indices.tolist()} (probs: {[f'{p:.3f}' for p in top3.values.tolist()]})")
    
    # Compare distributions
    print(f"\n🔍 Distribution Comparison:")
    print("-" * 40)
    
    if masked_pos.any():
        subs_dist = subs_probs[first_masked]
        score_dist = score_probs[first_masked]
        
        # KL divergence
        kl_div = F.kl_div(torch.log(score_dist + 1e-8), subs_dist, reduction='sum').item()
        print(f"   KL divergence (Score || SUBS): {kl_div:.3f}")
        
        # Jensen-Shannon divergence
        m = 0.5 * (subs_dist + score_dist)
        js_div = 0.5 * F.kl_div(torch.log(subs_dist + 1e-8), m, reduction='sum').item() + \
                 0.5 * F.kl_div(torch.log(score_dist + 1e-8), m, reduction='sum').item()
        print(f"   JS divergence: {js_div:.3f}")
        
        # Overlap (how much probability mass is shared)
        overlap = torch.sum(torch.min(subs_dist, score_dist)).item()
        print(f"   Probability overlap: {overlap:.3f}")
    
    return subs_probs, score_probs

def demonstrate_sampling_fix():
    """Demonstrate the sampling fix in action."""
    
    print(f"\n🔧 Sampling Fix Demonstration")
    print("=" * 60)
    
    print(f"🚨 The Problem:")
    print(f"   • Training: model(x, sigma, use_subs=True) → log probabilities")
    print(f"   • Sampling: model(x, sigma, use_subs=False) → raw scores")
    print(f"   • Result: Model learns one thing, uses another!")
    
    print(f"\n✅ The Solution:")
    print(f"   • Training: model(x, sigma, use_subs=True) → log probabilities")
    print(f"   • Sampling: model(x, sigma, use_subs=True) → log probabilities")
    print(f"   • Result: Consistent parameterization!")
    
    print(f"\n📈 Expected Benefits:")
    print(f"   • Better ESM perplexities (sequences follow learned distributions)")
    print(f"   • More realistic protein sequences")
    print(f"   • Improved training-sampling consistency")
    print(f"   • Better convergence and quality")

def test_esm_perplexity_prediction():
    """Predict how ESM perplexities should improve."""
    
    print(f"\n🔮 ESM Perplexity Prediction")
    print("=" * 60)
    
    print(f"🎯 Why ESM perplexities weren't improving:")
    print(f"   • Model trained to output good log probabilities (SUBS)")
    print(f"   • But sampling used raw scores (different distributions)")
    print(f"   • Generated sequences didn't follow learned protein patterns")
    print(f"   • ESM (trained on real proteins) gave high perplexities")
    
    print(f"\n✨ Why they should improve now:")
    print(f"   • Sampling now uses the same parameterization as training")
    print(f"   • Generated sequences follow learned protein distributions")
    print(f"   • Model's learned knowledge about protein patterns is actually used")
    print(f"   • ESM should give lower perplexities for more realistic sequences")
    
    print(f"\n📊 What to expect:")
    print(f"   • Gradual improvement in ESM perplexities over training steps")
    print(f"   • More diverse and realistic amino acid distributions")
    print(f"   • Better sequence quality metrics")
    print(f"   • Faster convergence to good protein-like sequences")

def main():
    """Run all tests."""
    
    print("🔬 SUBS vs Score-based Sampling Analysis")
    print("=" * 80)
    print("This script demonstrates the critical fix for sampling parameterization")
    print("that should resolve the ESM perplexity improvement issue.")
    print("=" * 80)
    
    # Run tests
    test_sampling_comparison()
    demonstrate_sampling_fix()
    test_esm_perplexity_prediction()
    
    print(f"\n🎯 Summary:")
    print("=" * 60)
    print("The sampling fix changes one line in the code but has huge impact:")
    print("   OLD: model(x, sigma, use_subs=False)  # Score-based")
    print("   NEW: model(x, sigma, use_subs=True)   # SUBS (consistent with training)")
    print("\nThis should dramatically improve ESM perplexities and sequence quality!")
    
    return True

if __name__ == "__main__":
    main()
