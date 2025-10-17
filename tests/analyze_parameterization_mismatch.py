#!/usr/bin/env python3
"""
Analyze the parameterization mismatch between training and sampling.

The issue: 
- Training uses SUBS parameterization (use_subs=True) → log probabilities
- Sampling uses score-based parameterization (use_subs=False) → raw logits/scores

This mismatch means the model learns one thing but samples with another!
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

def analyze_parameterization_difference():
    """Analyze the difference between SUBS and score-based parameterizations."""
    
    print("🔍 Analyzing Parameterization Mismatch")
    print("=" * 60)
    
    # Create mock model outputs
    batch_size, seq_len, vocab_size = 2, 10, 26
    mask_index = vocab_size - 1  # 25
    
    # Raw logits from model
    raw_logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Mock input tokens (some masked, some not)
    xt = torch.randint(0, vocab_size, (batch_size, seq_len))
    xt[0, :3] = mask_index  # First 3 positions are masked
    xt[1, 5:8] = mask_index  # Positions 5-8 are masked
    
    print(f"📊 Input Analysis:")
    print(f"   Raw logits shape: {raw_logits.shape}")
    print(f"   Input tokens (xt): {xt}")
    print(f"   Mask index: {mask_index}")
    print(f"   Masked positions: {(xt == mask_index).sum().item()}/{xt.numel()}")
    
    # 1. SUBS Parameterization (used in training)
    print(f"\n🎯 SUBS Parameterization (Training):")
    print("-" * 40)
    
    subs_logits = raw_logits.clone()
    neg_infinity = -1000000.0
    
    # Step 1: Set mask token log prob to -infinity
    subs_logits[:, :, mask_index] = neg_infinity
    
    # Step 2: Normalize to log probabilities
    subs_log_probs = subs_logits - torch.logsumexp(subs_logits, dim=-1, keepdim=True)
    
    # Step 3: For unmasked positions, set all to -inf except original token
    unmasked_mask = (xt != mask_index)
    subs_log_probs[unmasked_mask] = neg_infinity
    # Set original token log prob to 0 (prob = 1) for unmasked positions
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
    seq_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    subs_log_probs[batch_indices[unmasked_mask], seq_indices[unmasked_mask], xt[unmasked_mask]] = 0.0
    
    print(f"   SUBS log probs shape: {subs_log_probs.shape}")
    print(f"   SUBS log prob range: [{subs_log_probs[subs_log_probs > -999999].min().item():.3f}, {subs_log_probs[subs_log_probs > -999999].max().item():.3f}]")
    print(f"   Log prob sums: {torch.logsumexp(subs_log_probs, dim=-1).mean().item():.6f} (should be ~0)")
    
    # Convert to probabilities for analysis
    subs_probs = torch.exp(subs_log_probs)
    print(f"   SUBS prob range: [{subs_probs.min().item():.6f}, {subs_probs.max().item():.6f}]")
    print(f"   Prob sums: {subs_probs.sum(dim=-1).mean().item():.6f} (should be ~1)")
    
    # 2. Score-based Parameterization (used in sampling)
    print(f"\n⚡ Score-based Parameterization (Sampling):")
    print("-" * 40)
    
    score_logits = raw_logits.clone()
    
    # Apply absorbing state logic (zero out positions that match input)
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
    seq_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    indices_expanded = xt.unsqueeze(-1)  # [batch, seq, 1]
    zeros = torch.zeros_like(score_logits[..., :1])  # [batch, seq, 1]
    score_logits = torch.scatter(score_logits, -1, indices_expanded, zeros)
    
    print(f"   Score logits shape: {score_logits.shape}")
    print(f"   Score logits range: [{score_logits.min().item():.3f}, {score_logits.max().item():.3f}]")
    
    # Convert to probabilities for comparison
    score_probs = F.softmax(score_logits, dim=-1)
    print(f"   Score prob range: [{score_probs.min().item():.6f}, {score_probs.max().item():.6f}]")
    print(f"   Prob sums: {score_probs.sum(dim=-1).mean().item():.6f} (should be ~1)")
    
    # 3. Analyze the mismatch
    print(f"\n🚨 Parameterization Mismatch Analysis:")
    print("-" * 40)
    
    # For masked positions, compare the distributions
    masked_positions = (xt == mask_index)
    
    if masked_positions.any():
        # Get distributions for first masked position
        first_masked_batch, first_masked_seq = torch.where(masked_positions)[0][0], torch.where(masked_positions)[1][0]
        
        subs_dist = subs_probs[first_masked_batch, first_masked_seq]
        score_dist = score_probs[first_masked_batch, first_masked_seq]
        
        print(f"   Comparing distributions for masked position [{first_masked_batch}, {first_masked_seq}]:")
        print(f"   SUBS distribution entropy: {-torch.sum(subs_dist * torch.log(subs_dist + 1e-8)).item():.3f}")
        print(f"   Score distribution entropy: {-torch.sum(score_dist * torch.log(score_dist + 1e-8)).item():.3f}")
        
        # KL divergence between distributions
        kl_div = F.kl_div(torch.log(score_dist + 1e-8), subs_dist, reduction='sum').item()
        print(f"   KL divergence (Score || SUBS): {kl_div:.3f}")
        
        # Show top-5 tokens for each
        subs_top5 = torch.topk(subs_dist, 5)
        score_top5 = torch.topk(score_dist, 5)
        
        print(f"   SUBS top-5 tokens: {subs_top5.indices.tolist()} (probs: {subs_top5.values.tolist()})")
        print(f"   Score top-5 tokens: {score_top5.indices.tolist()} (probs: {score_top5.values.tolist()})")
    
    return subs_log_probs, score_logits

def demonstrate_sampling_fix():
    """Demonstrate how to fix the sampling to use SUBS parameterization."""
    
    print(f"\n🔧 Proposed Fix: Use SUBS in Sampling")
    print("=" * 60)
    
    print(f"Current sampling flow:")
    print(f"   1. model(x, sigma, use_subs=False) → raw logits/scores")
    print(f"   2. mutils.get_score_fn() → torch.exp(output) → probabilities")
    print(f"   3. graph.staggered_score() → sampling probabilities")
    print(f"   4. sample_categorical() → next tokens")
    
    print(f"\nProposed sampling flow:")
    print(f"   1. model(x, sigma, use_subs=True) → log probabilities")
    print(f"   2. mutils.get_score_fn() → torch.exp(output) → probabilities")
    print(f"   3. graph.staggered_score() → sampling probabilities")
    print(f"   4. sample_categorical() → next tokens")
    
    print(f"\n✅ Benefits of using SUBS in sampling:")
    print(f"   • Training and sampling use the same parameterization")
    print(f"   • Model learns what it actually uses during generation")
    print(f"   • Better sequence quality and ESM perplexities")
    print(f"   • Consistent probability distributions")
    
    print(f"\n⚠️  Potential concerns:")
    print(f"   • Need to verify graph operations work with SUBS probabilities")
    print(f"   • May need to adjust sampling hyperparameters")
    print(f"   • Should test on small examples first")

def analyze_current_training_vs_sampling():
    """Analyze what the model learns vs what it uses."""
    
    print(f"\n📚 Training vs Sampling Analysis")
    print("=" * 60)
    
    print(f"🎓 What the model learns (Training):")
    print(f"   • SUBS parameterization with use_subs=True")
    print(f"   • Log probabilities with specific constraints:")
    print(f"     - Mask token has -∞ log probability")
    print(f"     - Unmasked positions have deterministic distributions")
    print(f"     - Masked positions have learned distributions over non-mask tokens")
    print(f"   • Loss function: SUBS loss on log probabilities")
    
    print(f"\n🎯 What the model uses (Sampling):")
    print(f"   • Score-based parameterization with use_subs=False")
    print(f"   • Raw logits/scores with absorbing state logic:")
    print(f"     - Positions matching input get zeroed out")
    print(f"     - No special handling of mask token")
    print(f"     - Different probability distributions")
    print(f"   • Sampling: Convert scores to probabilities via softmax")
    
    print(f"\n🚨 The Problem:")
    print(f"   • Model is trained to output log probabilities (SUBS)")
    print(f"   • But sampling treats outputs as raw scores")
    print(f"   • This creates a fundamental mismatch!")
    print(f"   • ESM perplexities don't improve because generated sequences")
    print(f"     don't follow the learned probability distributions")
    
    print(f"\n💡 The Solution:")
    print(f"   • Use use_subs=True during sampling")
    print(f"   • This makes training and sampling consistent")
    print(f"   • Model generates sequences using what it actually learned")

def main():
    """Run the analysis."""
    
    print("🔬 Parameterization Mismatch Analysis")
    print("=" * 80)
    print("This script analyzes the critical mismatch between training and sampling")
    print("parameterizations that may be causing poor ESM perplexity improvement.")
    print("=" * 80)
    
    # Run analyses
    analyze_parameterization_difference()
    demonstrate_sampling_fix()
    analyze_current_training_vs_sampling()
    
    print(f"\n🎯 Conclusion:")
    print("=" * 60)
    print("The lack of ESM perplexity improvement is likely due to the mismatch")
    print("between SUBS parameterization (training) and score-based parameterization (sampling).")
    print("\nRecommendation: Modify sampling to use use_subs=True for consistency.")
    
    return True

if __name__ == "__main__":
    main()
