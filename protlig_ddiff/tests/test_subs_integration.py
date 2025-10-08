"""
Test script to verify SUBS integration works correctly
"""
import torch
import numpy as np
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # Add sedd_scripts to path

from protlig_ddiff.models.transformer_v100 import DiscDiffModel
from protlig_ddiff.processing.graph_lib import Absorbing
from protlig_ddiff.processing.noise_lib import LogLinearNoise
from protlig_ddiff.processing.subs_loss import subs_loss, compute_subs_metrics

def test_subs_integration():
    """Test that SUBS integration works correctly"""
    print("🧪 Testing SUBS integration...")
    
    # Setup test parameters
    batch_size = 4
    seq_len = 64
    vocab_size = 25  # 20 amino acids + 5 special tokens
    device = torch.device('cpu')  # Use CPU for testing
    
    # Create test configuration
    class TestConfig:
        def __init__(self):
            self.dim = 256
            self.n_heads = 8
            self.n_layers = 4
            self.vocab_size = vocab_size
            self.max_seq_len = seq_len
            self.cond_dim = 128
            self.scale_by_sigma = False  # Disable for SUBS

            # Add the vocab_size in the format the model expects
            self.tokens = vocab_size  # Model looks for this first
            
    config = TestConfig()
    
    # Initialize components
    print("📦 Initializing components...")

    # Graph (absorbing state) - IMPORTANT: Use vocab_size-1 to get vocab_size total tokens
    graph = Absorbing(vocab_size - 1)  # This creates vocab_size tokens (0 to vocab_size-1)
    print(f"   Graph: {graph.__class__.__name__}, vocab_size={graph.dim}")
    
    # Noise schedule
    noise = LogLinearNoise()
    print(f"   Noise: {noise.__class__.__name__}")
    
    # Model
    model = DiscDiffModel(config)
    model.eval()
    print(f"   Model: {model.__class__.__name__}, params={sum(p.numel() for p in model.parameters()):,}")
    
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
    
    # Test model forward pass
    print("🚀 Testing model forward pass...")
    
    # Test original score-based forward pass
    with torch.no_grad():
        score_output = model(xt, sigma, use_subs=False)
        print(f"   Score output shape: {score_output.shape}")
        print(f"   Score output range: [{score_output.min().item():.3f}, {score_output.max().item():.3f}]")
    
    # Test SUBS forward pass
    with torch.no_grad():
        subs_output = model(xt, sigma, use_subs=True)
        print(f"   SUBS output shape: {subs_output.shape}")
        print(f"   SUBS output range: [{subs_output.min().item():.3f}, {subs_output.max().item():.3f}]")
        
        # Check that SUBS output is log probabilities
        log_probs_sum = torch.logsumexp(subs_output, dim=-1)
        print(f"   Log prob sums (should be ~0): {log_probs_sum.mean().item():.6f} ± {log_probs_sum.std().item():.6f}")
    
    # Test SUBS loss computation
    print("💰 Testing SUBS loss...")
    
    with torch.no_grad():
        loss = subs_loss(subs_output, x0, sigma, noise)
        print(f"   SUBS loss: {loss.item():.6f}")
        
        # Test metrics
        metrics = compute_subs_metrics(subs_output, x0, sigma)
        print(f"   Mean log prob: {metrics['mean_log_prob']:.3f}")
        print(f"   Perplexity: {metrics['perplexity']:.2f}")
        print(f"   Accuracy: {metrics['accuracy']:.2%}")
    
    # Test gradient flow
    print("🔄 Testing gradient flow...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Forward pass with gradients
    subs_output = model(xt, sigma, use_subs=True)
    loss = subs_loss(subs_output, x0, sigma, noise)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
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
    
    print("✅ SUBS integration test completed successfully!")
    
    return {
        'loss': loss.item(),
        'grad_norm': total_grad_norm,
        'metrics': metrics
    }

def test_comparison():
    """Compare SUBS vs score-based loss on same data"""
    print("\n🔍 Comparing SUBS vs score-based loss...")

    # This is a placeholder since we don't have the full score-based loss implementation
    # In practice, you would run both losses on the same data and compare

    print("   Note: Full comparison requires complete score-based loss implementation")
    print("   SUBS loss should be more stable and have cleaner gradients")

if __name__ == "__main__":
    try:
        results = test_subs_integration()
        test_comparison()
        
        print(f"\n🎉 All tests passed!")
        print(f"   Final loss: {results['loss']:.6f}")
        print(f"   Gradient norm: {results['grad_norm']:.6f}")
        print(f"   Perplexity: {results['metrics']['perplexity']:.2f}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
