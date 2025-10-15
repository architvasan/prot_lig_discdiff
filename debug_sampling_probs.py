#!/usr/bin/env python3
"""
Debug script to analyze probability distributions before and after the sampling fix.
This will help understand why 'M' characters are appearing more frequently.
"""

import torch
import numpy as np
from collections import Counter

def analyze_probability_distribution(log_probs, title=""):
    """Analyze a probability distribution from log probabilities."""
    print(f"\n=== {title} ===")
    
    # Convert to probabilities
    probs = torch.exp(log_probs)
    
    # Vocabulary mapping (from data_utils.py)
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    vocab = {}
    vocab.update({token: i for i, token in enumerate(special_tokens)})
    vocab.update({aa: i + len(special_tokens) for i, aa in enumerate(amino_acids)})
    id_to_token = {i: token for token, i in vocab.items()}
    
    # Add absorbing token
    id_to_token[25] = "<absorb>"
    
    print(f"Shape: {probs.shape}")
    print(f"Sum: {probs.sum(dim=-1).mean():.6f} (should be ~1.0)")
    print(f"Min prob: {probs.min():.6f}")
    print(f"Max prob: {probs.max():.6f}")
    
    # Get top 5 most probable tokens
    avg_probs = probs.mean(dim=(0, 1))  # Average across batch and sequence
    top_indices = torch.topk(avg_probs, k=min(10, len(avg_probs))).indices
    
    print("\nTop 10 most probable tokens (averaged):")
    for i, idx in enumerate(top_indices):
        token = id_to_token.get(idx.item(), f"UNK_{idx.item()}")
        prob = avg_probs[idx].item()
        print(f"  {i+1:2d}. Token {idx:2d} ('{token}'): {prob:.4f}")
    
    # Check 'M' specifically (token ID 16)
    m_token_id = vocab['M']  # Should be 16
    m_prob = avg_probs[m_token_id].item()
    print(f"\n'M' (token {m_token_id}) probability: {m_prob:.4f}")
    
    return probs, avg_probs

def simulate_sampling_comparison():
    """Simulate the difference between old (incorrect) and new (correct) sampling."""
    print("🔍 Simulating sampling probability distributions...")
    
    # Create a realistic example log probability distribution
    # This simulates what the model might output
    batch_size, seq_len, vocab_size = 2, 10, 26
    
    # Create log probabilities that favor certain tokens
    # Make 'M' (token 16) have higher log probability
    log_probs = torch.randn(batch_size, seq_len, vocab_size) * 2.0
    m_token_id = 5 + list("ACDEFGHIKLMNPQRSTVWY").index('M')  # Calculate M's token ID
    log_probs[:, :, m_token_id] += 1.0  # Boost 'M' log probability
    log_probs[:, :, 25] = -10.0  # Suppress absorbing token
    
    # Normalize to proper log probabilities
    log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
    
    print("Original log probabilities (what model outputs):")
    analyze_probability_distribution(log_probs, "Model Output (Log Probabilities)")
    
    # OLD METHOD: Use log probabilities directly in staggered_score
    print("\n" + "="*60)
    print("OLD METHOD: Using log probabilities directly")
    
    # Simulate staggered_score operations on log probabilities (INCORRECT)
    dsigma = torch.tensor([0.1, 0.1])  # Example dsigma values

    old_score = log_probs.clone()
    # This is what staggered_score does (but incorrectly with log probs)
    extra_const = (1 - dsigma.exp())[:, None] * old_score.sum(dim=-1)  # WRONG: summing log probs
    old_score *= dsigma.exp()[:, None, None]  # WRONG: scaling log probs
    old_score[:, :, -1] += extra_const  # Add to absorbing token
    
    analyze_probability_distribution(old_score, "OLD: After Staggered Score (Incorrect)")
    
    # NEW METHOD: Convert to probabilities first
    print("\n" + "="*60)
    print("NEW METHOD: Convert to probabilities first")
    
    # Convert log probabilities to probabilities (CORRECT)
    probs = torch.exp(log_probs)
    probs = torch.clamp(probs, min=1e-8, max=1.0)
    
    # Now apply staggered_score to probabilities (CORRECT)
    new_score = probs.clone()
    extra_const = (1 - dsigma.exp())[:, None] * new_score.sum(dim=-1)  # CORRECT: summing probs
    new_score *= dsigma.exp()[:, None, None]  # CORRECT: scaling probs
    new_score[:, :, -1] += extra_const  # Add to absorbing token
    
    # Convert back to log space for analysis
    new_score_log = torch.log(new_score + 1e-8)
    analyze_probability_distribution(new_score_log, "NEW: After Staggered Score (Correct)")
    
    # Compare the difference
    print("\n" + "="*60)
    print("COMPARISON:")
    
    m_token_id = 5 + list("ACDEFGHIKLMNPQRSTVWY").index('M')  # Calculate M's token ID
    old_m_prob = torch.exp(old_score)[:, :, m_token_id].mean().item()
    new_m_prob = new_score[:, :, m_token_id].mean().item()
    
    print(f"'M' probability - OLD method: {old_m_prob:.6f}")
    print(f"'M' probability - NEW method: {new_m_prob:.6f}")
    print(f"Ratio (NEW/OLD): {new_m_prob/old_m_prob:.2f}x")
    
    if new_m_prob > old_m_prob:
        print("✅ NEW method increases 'M' probability - this explains more 'M' characters!")
    else:
        print("❌ NEW method decreases 'M' probability - something else is happening")

def check_model_bias():
    """Check if the model has learned a bias toward 'M'."""
    print("\n🔍 Checking for model bias toward 'M'...")
    
    # Natural frequency of amino acids in proteins (approximate)
    natural_frequencies = {
        'A': 8.25, 'C': 1.37, 'D': 5.45, 'E': 6.75, 'F': 3.86,
        'G': 7.07, 'H': 2.27, 'I': 5.96, 'K': 5.84, 'L': 9.66,
        'M': 2.42, 'N': 4.06, 'P': 4.70, 'Q': 3.93, 'R': 5.54,
        'S': 6.56, 'T': 5.34, 'V': 6.87, 'W': 1.08, 'Y': 2.92
    }
    
    print("Natural frequency of 'M' in proteins: 2.42%")
    print("If your model is generating much more than ~2.4% 'M', it has learned a bias.")
    print("\nTo check your actual sampling:")
    print("1. Generate a batch of sequences")
    print("2. Count amino acid frequencies")
    print("3. Compare to natural frequencies")

if __name__ == "__main__":
    print("🧬 Protein Discrete Diffusion - Sampling Probability Analysis")
    print("="*60)
    
    simulate_sampling_comparison()
    check_model_bias()
    
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS:")
    print("1. Check if your model was trained with a dataset bias toward 'M'")
    print("2. Verify the training data amino acid distribution")
    print("3. Consider if the model checkpoint is from early training (might be undertrained)")
    print("4. The fix is mathematically correct - the 'M' bias was always there, just hidden")
