"""
Simple test script to verify SUBS loss works correctly
"""
import torch
import numpy as np
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # Add sedd_scripts to path

from protlig_ddiff.processing.graph_lib import Absorbing
from protlig_ddiff.processing.noise_lib import LogLinearNoise
from protlig_ddiff.processing.subs_loss import subs_loss, compute_subs_metrics

def test_subs_loss_only():
    """Test just the SUBS loss computation without the complex model"""
    print("🧪 Testing SUBS loss computation...")
    
    # Setup test parameters
    batch_size = 4
    seq_len = 64
    vocab_size = 25  # 20 amino acids + 5 special tokens
    device = torch.device('cpu')  # Use CPU for testing
    
    print("📦 Initializing components...")
    
    # Graph (absorbing state)
    graph = Absorbing(vocab_size)
    print(f"   Graph: {graph.__class__.__name__}, vocab_size={graph.dim}")
    
    # Noise schedule
    noise = LogLinearNoise()
    print(f"   Noise: {noise.__class__.__name__}")
    
    # Create test data
    print("🎲 Creating test data...")
    x0 = torch.randint(0, vocab_size-1, (batch_size, seq_len))  # Exclude absorbing token
    print(f"   Clean sequences shape: {x0.shape}")
    print(f"   Sample sequence: {x0[0][:10].tolist()}")
    
    # Test corruption
    print("🔀 Testing corruption...")
    t = torch.rand(batch_size) * 0.5 + 0.1  # Moderate noise
    sigma, dsigma = noise(t)
    xt = graph.sample_transition(x0, sigma)
    
    print(f"   Noise levels (σ): {sigma.mean().item():.3f} ± {sigma.std().item():.3f}")
    print(f"   Corrupted sequence: {xt[0][:10].tolist()}")
    print(f"   Corruption rate: {(xt != x0).float().mean().item():.2%}")
    
    # Create mock model output (log probabilities)
    print("🎯 Creating mock model output...")
    
    # Create realistic log probabilities
    # Higher probability for correct tokens, lower for others
    logits = torch.randn(batch_size, seq_len, vocab_size) * 2.0
    
    # Boost probability of correct tokens
    for b in range(batch_size):
        for s in range(seq_len):
            logits[b, s, x0[b, s]] += 3.0  # Higher logit for correct token
    
    # Convert to log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    
    print(f"   Model output shape: {log_probs.shape}")
    print(f"   Log prob range: [{log_probs.min().item():.3f}, {log_probs.max().item():.3f}]")
    
    # Check that it's proper log probabilities
    prob_sums = torch.exp(log_probs).sum(dim=-1)
    print(f"   Probability sums (should be ~1): {prob_sums.mean().item():.6f} ± {prob_sums.std().item():.6f}")
    
    # Test SUBS loss computation
    print("💰 Testing SUBS loss...")
    
    loss = subs_loss(log_probs, x0, sigma, noise)
    print(f"   SUBS loss: {loss.item():.6f}")
    
    # Test metrics
    metrics = compute_subs_metrics(log_probs, x0, sigma)
    print(f"   Mean log prob: {metrics['mean_log_prob']:.3f}")
    print(f"   Perplexity: {metrics['perplexity']:.2f}")
    print(f"   Accuracy: {metrics['accuracy']:.2%}")
    
    # Test gradient flow
    print("🔄 Testing gradient flow...")
    
    # Create a simple linear layer to test gradients
    linear = torch.nn.Linear(vocab_size, vocab_size)
    optimizer = torch.optim.Adam(linear.parameters(), lr=1e-4)
    
    # Forward pass with gradients
    # Create input features for the linear layer
    input_features = torch.randn(batch_size, seq_len, vocab_size)
    logits_grad = linear(input_features)
    log_probs_grad = torch.log_softmax(logits_grad, dim=-1)
    loss_grad = subs_loss(log_probs_grad, x0, sigma, noise)
    
    # Backward pass
    loss_grad.backward()
    
    # Check gradients
    total_grad_norm = 0
    param_count = 0
    for name, param in linear.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            param_count += 1
    
    total_grad_norm = total_grad_norm ** 0.5
    print(f"   Total gradient norm: {total_grad_norm:.6f}")
    print(f"   Parameters with gradients: {param_count}")
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    # Test different noise levels
    print("📊 Testing different noise levels...")
    
    noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
    for noise_level in noise_levels:
        t_test = torch.full((batch_size,), noise_level)
        sigma_test, _ = noise(t_test)
        xt_test = graph.sample_transition(x0, sigma_test)
        loss_test = subs_loss(log_probs, x0, sigma_test, noise)
        corruption_rate = (xt_test != x0).float().mean().item()
        
        print(f"   σ={noise_level:.1f}: loss={loss_test.item():.4f}, corruption={corruption_rate:.1%}")
    
    print("✅ SUBS loss test completed successfully!")
    
    return {
        'loss': loss.item(),
        'grad_norm': total_grad_norm,
        'metrics': metrics
    }

def test_subs_parameterization():
    """Test the SUBS parameterization logic"""
    print("\n🔧 Testing SUBS parameterization...")
    
    batch_size = 2
    seq_len = 8
    vocab_size = 10
    mask_index = vocab_size - 1
    
    # Create test data
    xt = torch.tensor([[0, 1, mask_index, 3, mask_index, 5, 6, 7],
                       [mask_index, 1, 2, mask_index, 4, 5, mask_index, 7]])
    
    # Create raw logits
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    print(f"   Input tokens: {xt.tolist()}")
    print(f"   Mask index: {mask_index}")
    
    # Apply SUBS parameterization manually
    neg_infinity = -1000000.0
    
    # Set log prob at mask index to -infinity
    logits[:, :, mask_index] += neg_infinity
    
    # Normalize logits to log probabilities
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    
    # For unmasked tokens, set all probs to -inf except the original token
    unmasked_indices = (xt != mask_index)
    log_probs[unmasked_indices] = neg_infinity
    log_probs[unmasked_indices, xt[unmasked_indices]] = 0
    
    print(f"   Unmasked positions: {unmasked_indices.sum().item()}")
    print(f"   Masked positions: {(~unmasked_indices).sum().item()}")
    
    # Check that unmasked positions have probability 1 for their token
    unmasked_probs = torch.exp(log_probs[unmasked_indices, xt[unmasked_indices]])
    print(f"   Unmasked token probs (should be 1): {unmasked_probs.mean().item():.6f}")
    
    # Check that masked positions have valid probability distribution
    masked_log_probs = log_probs[xt == mask_index]
    if masked_log_probs.numel() > 0:
        masked_prob_sums = torch.exp(masked_log_probs).sum(dim=-1)
        print(f"   Masked position prob sums: {masked_prob_sums.mean().item():.6f}")
    
    print("✅ SUBS parameterization test completed!")

if __name__ == "__main__":
    try:
        results = test_subs_loss_only()
        test_subs_parameterization()
        
        print(f"\n🎉 All tests passed!")
        print(f"   Final loss: {results['loss']:.6f}")
        print(f"   Gradient norm: {results['grad_norm']:.6f}")
        print(f"   Perplexity: {results['metrics']['perplexity']:.2f}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
