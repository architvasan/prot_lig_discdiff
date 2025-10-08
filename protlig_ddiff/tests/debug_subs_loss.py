"""
Debug SUBS loss computation to find why it's so high
"""
import torch
import numpy as np
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # Add sedd_scripts to path

from protlig_ddiff.processing.noise_lib import LogLinearNoise
from protlig_ddiff.processing.subs_loss import subs_loss

def debug_subs_loss():
    """Debug SUBS loss computation"""
    print("🔍 Debugging SUBS loss...")
    
    # Setup test parameters
    batch_size = 2
    seq_len = 8
    vocab_size = 25
    
    # Create test data
    x0 = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Ground truth tokens: {x0}")
    
    # Create realistic model output (log probabilities)
    logits = torch.randn(batch_size, seq_len, vocab_size) * 2.0
    log_probs = torch.log_softmax(logits, dim=-1)
    print(f"Log prob range: [{log_probs.min().item():.3f}, {log_probs.max().item():.3f}]")
    
    # Extract log probs of ground truth tokens
    log_p_theta = torch.gather(log_probs, dim=-1, index=x0[:, :, None]).squeeze(-1)
    print(f"Ground truth log probs: {log_p_theta}")
    print(f"Ground truth log prob range: [{log_p_theta.min().item():.3f}, {log_p_theta.max().item():.3f}]")
    
    # Test noise schedule
    noise = LogLinearNoise()
    
    # Test different TIME values (t should be in [0,1])
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t_val in t_values:
        print(f"\n--- Time t = {t_val} ---")

        t = torch.full((batch_size, 1), t_val)  # [batch, 1] format

        # Get sigma and dsigma from time
        sigma, dsigma = noise(t)
        print(f"σ(t): {sigma.flatten()}")
        print(f"dσ/dt: {dsigma.flatten()}")

        # Compute weight
        expm1_sigma = torch.expm1(sigma)
        print(f"exp(σ) - 1: {expm1_sigma.flatten()}")

        weight = dsigma / expm1_sigma
        print(f"weight (dσ/dt)/(exp(σ)-1): {weight.flatten()}")

        # Compute loss per token
        loss_per_token = -log_p_theta * weight
        print(f"loss per token range: [{loss_per_token.min().item():.3f}, {loss_per_token.max().item():.3f}]")
        
        # Total loss
        total_loss = loss_per_token.mean()
        print(f"Total loss: {total_loss.item():.6f}")
        
        # Compare with simple NLL
        simple_nll = -log_p_theta.mean()
        print(f"Simple NLL (no weighting): {simple_nll.item():.6f}")
        print(f"Weight amplification factor: {(total_loss / simple_nll).item():.2f}x")

def debug_noise_schedule():
    """Debug the noise schedule itself"""
    print("\n🔍 Debugging noise schedule...")
    
    noise = LogLinearNoise()
    
    # Test the noise schedule implementation
    t_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for t in t_values:
        print(f"\n--- t = {t} ---")
        
        t_tensor = torch.tensor([[t]])  # [1, 1] shape
        
        try:
            sigma, dsigma = noise(t_tensor)
            print(f"σ(t): {sigma.item():.6f}")
            print(f"dσ/dt: {dsigma.item():.6f}")
            
            # Check the mathematical relationship
            # For LogLinear: σ(t) = -log(1 - (1-ε)t)
            # dσ/dt = (1-ε) / (1 - (1-ε)t)
            epsilon = 1e-3  # Default epsilon
            expected_sigma = -torch.log(1 - (1-epsilon) * t)
            expected_dsigma = (1-epsilon) / (1 - (1-epsilon) * t)
            
            print(f"Expected σ: {expected_sigma.item():.6f}")
            print(f"Expected dσ/dt: {expected_dsigma.item():.6f}")
            
            # Check expm1
            expm1_val = torch.expm1(sigma)
            print(f"exp(σ) - 1: {expm1_val.item():.6f}")
            
            # Check the weight
            weight = dsigma / expm1_val
            print(f"Weight (dσ/dt)/(exp(σ)-1): {weight.item():.6f}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    debug_subs_loss()
    debug_noise_schedule()
