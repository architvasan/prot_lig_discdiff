"""
Detailed debug of SUBS parameterization and loss
"""
import torch
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # Add sedd_scripts to path

from protlig_ddiff.models.transformer_v100 import DiscDiffModel
from protlig_ddiff.processing.graph_lib import Absorbing
from protlig_ddiff.processing.noise_lib import LogLinearNoise
from protlig_ddiff.processing.subs_loss import subs_loss

def debug_subs_detailed():
    """Debug SUBS parameterization in detail"""
    print("🔍 Detailed SUBS debug...")
    
    # Simple test case
    batch_size = 2
    seq_len = 4
    vocab_size = 6  # Small vocab for easier debugging
    
    # Create test configuration
    class TestConfig:
        def __init__(self):
            self.dim = 128
            self.n_heads = 4
            self.n_layers = 1
            self.vocab_size = vocab_size  # This should match the graph
            self.max_seq_len = seq_len
            self.cond_dim = 64
            self.scale_by_sigma = False

        # Also add these attributes that the model might look for
        @property
        def model(self):
            return self
            
    config = TestConfig()
    
    # Initialize components
    # IMPORTANT: Absorbing(n) creates n+1 tokens, so use vocab_size-1
    graph = Absorbing(vocab_size - 1)  # This will create vocab_size tokens total
    noise = LogLinearNoise()
    model = DiscDiffModel(config)

    print(f"Config vocab_size: {vocab_size}")
    print(f"Model vocab_size: {model.vocab_size}")
    print(f"Graph dim: {graph.dim} (should equal vocab_size={vocab_size})")
    print(f"Absorbing token: {graph.dim - 1} (should be {vocab_size - 1})")
    print(f"Model mask token: {model.vocab_size - 1}")
    model.eval()
    
    # Create test data
    x0 = torch.tensor([[0, 1, 2, 3], [4, 0, 1, 2]])  # Ground truth
    print(f"Ground truth x0: {x0}")
    
    # Test corruption with higher noise to ensure masking
    t = torch.tensor([0.8, 0.9])  # Higher noise levels to force masking
    sigma, dsigma = noise(t.unsqueeze(-1))
    print(f"Time t: {t}")
    print(f"Sigma: {sigma.flatten()}")
    print(f"dSigma/dt: {dsigma.flatten()}")
    
    # Corrupt the data
    xt = graph.sample_transition(x0, sigma)
    print(f"Corrupted xt: {xt}")
    print(f"xt shape: {xt.shape}")
    print(f"Mask token (absorbing): {vocab_size-1}")

    # Fix shape if needed
    if xt.dim() != 2:
        print(f"Fixing xt shape from {xt.shape} to 2D")
        xt = xt.view(batch_size, -1)
        print(f"Fixed xt: {xt}")

    # Check which tokens are masked
    mask_token = vocab_size - 1
    masked_positions = (xt == mask_token)
    print(f"Masked positions: {masked_positions}")
    print(f"Number of masked tokens: {masked_positions.sum().item()}")
    
    # Test model forward pass
    with torch.no_grad():
        # Get raw logits
        raw_output = model(xt, sigma.flatten(), use_subs=False)
        print(f"\nRaw model output shape: {raw_output.shape}")
        print(f"Raw output range: [{raw_output.min().item():.3f}, {raw_output.max().item():.3f}]")

        # Check raw logits for specific positions
        print(f"\n--- Raw logits analysis ---")
        for b in range(batch_size):
            for s in range(seq_len):
                gt_token = x0[b, s].item()
                curr_token = xt[b, s].item()
                is_masked = (curr_token == mask_token)

                raw_logits = raw_output[b, s]
                gt_raw_logit = raw_logits[gt_token].item()
                mask_raw_logit = raw_logits[mask_token].item()

                print(f"Batch {b}, Seq {s}: GT={gt_token}, Current={curr_token}, Masked={is_masked}")
                print(f"  Raw GT logit: {gt_raw_logit:.3f}, Raw mask logit: {mask_raw_logit:.3f}")
                print(f"  Raw logit range: [{raw_logits.min().item():.3f}, {raw_logits.max().item():.3f}]")
        
        # Get SUBS parameterized output
        subs_output = model(xt, sigma.flatten(), use_subs=True)
        print(f"\nSUBS output shape: {subs_output.shape}")
        print(f"SUBS output range: [{subs_output.min().item():.3f}, {subs_output.max().item():.3f}]")
        
        # Check log prob normalization
        log_prob_sums = torch.logsumexp(subs_output, dim=-1)
        print(f"Log prob sums: {log_prob_sums}")
        
        # Examine specific positions
        print(f"\n--- Detailed position analysis ---")
        for b in range(batch_size):
            for s in range(seq_len):
                gt_token = x0[b, s].item()
                curr_token = xt[b, s].item()
                is_masked = (curr_token == mask_token)
                
                log_probs = subs_output[b, s]
                gt_log_prob = log_probs[gt_token].item()
                
                print(f"Batch {b}, Seq {s}: GT={gt_token}, Current={curr_token}, Masked={is_masked}")
                print(f"  GT log prob: {gt_log_prob:.6f}")
                print(f"  All log probs: {log_probs.tolist()}")
        
        # Compute SUBS loss step by step
        print(f"\n--- SUBS loss computation ---")
        
        # Extract log probabilities of ground truth tokens
        log_p_theta = torch.gather(subs_output, dim=-1, index=x0[:, :, None]).squeeze(-1)
        print(f"Ground truth log probs: {log_p_theta}")
        
        # Compute weights
        weight = dsigma / torch.expm1(sigma)
        print(f"Weights: {weight.flatten()}")
        
        # Compute loss per token
        loss_per_token = -log_p_theta * weight
        print(f"Loss per token: {loss_per_token}")
        
        # Total loss
        total_loss = loss_per_token.mean()
        print(f"Total loss: {total_loss.item():.6f}")
        
        # Compare with simple NLL
        simple_nll = -log_p_theta.mean()
        print(f"Simple NLL: {simple_nll.item():.6f}")
        print(f"Weight amplification: {(total_loss / simple_nll).item():.2f}x")

if __name__ == "__main__":
    debug_subs_detailed()
