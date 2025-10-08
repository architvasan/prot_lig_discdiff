"""
Debug normalization with -infinity values
"""
import torch

def test_normalization():
    """Test what happens when we normalize logits with -infinity"""
    print("🔍 Testing normalization with -infinity...")
    
    # Create simple test logits
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 tokens
    print(f"Original logits: {logits}")
    
    # Normal normalization
    normal_log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    print(f"Normal log probs: {normal_log_probs}")
    print(f"Normal probs: {torch.exp(normal_log_probs)}")
    print(f"Sum of normal probs: {torch.exp(normal_log_probs).sum()}")
    
    print("\n" + "="*50)
    
    # Set one token to -infinity and normalize
    logits_with_inf = logits.clone()
    logits_with_inf[4] = -1000000.0  # Set last token to -inf
    print(f"Logits with -inf: {logits_with_inf}")
    
    inf_log_probs = logits_with_inf - torch.logsumexp(logits_with_inf, dim=-1, keepdim=True)
    print(f"Log probs after norm: {inf_log_probs}")
    print(f"Probs after norm: {torch.exp(inf_log_probs)}")
    print(f"Sum of probs: {torch.exp(inf_log_probs).sum()}")
    
    print("\n" + "="*50)
    
    # What if we set -inf AFTER normalization?
    logits_reset = logits.clone()
    normal_log_probs_2 = logits_reset - torch.logsumexp(logits_reset, dim=-1, keepdim=True)
    normal_log_probs_2[4] = -1000000.0  # Set to -inf AFTER normalization
    print(f"Set -inf AFTER normalization: {normal_log_probs_2}")
    print(f"Probs: {torch.exp(normal_log_probs_2)}")
    print(f"Sum of probs: {torch.exp(normal_log_probs_2).sum()}")
    
    print("\n" + "="*50)
    
    # Test the SUBS approach: exclude mask token from normalization
    print("Testing SUBS approach...")
    
    # Step 1: Set mask token to -inf
    mask_index = 4
    logits_subs = logits.clone()
    logits_subs[mask_index] = -1000000.0
    print(f"Step 1 - Set mask to -inf: {logits_subs}")
    
    # Step 2: Normalize (this should work correctly)
    log_probs_subs = logits_subs - torch.logsumexp(logits_subs, dim=-1, keepdim=True)
    print(f"Step 2 - After normalization: {log_probs_subs}")
    print(f"Probs: {torch.exp(log_probs_subs)}")
    print(f"Sum: {torch.exp(log_probs_subs).sum()}")
    
    # The issue might be that logsumexp(-inf) is problematic
    print(f"logsumexp of original: {torch.logsumexp(logits, dim=-1)}")
    print(f"logsumexp with -inf: {torch.logsumexp(logits_subs, dim=-1)}")

if __name__ == "__main__":
    test_normalization()
