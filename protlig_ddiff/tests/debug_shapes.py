"""
Debug script to find where shape corruption starts
"""
import torch
import sys
import os
# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # Add sedd_scripts to path

from protlig_ddiff.models.transformer_v100 import DiscDiffModel

def debug_shapes():
    """Debug where shape corruption starts"""
    print("🔍 Debugging shape corruption...")
    
    # Setup test parameters
    batch_size = 2  # Smaller for easier debugging
    seq_len = 8
    vocab_size = 25
    device = torch.device('cpu')
    
    # Create test configuration
    class TestConfig:
        def __init__(self):
            self.dim = 256
            self.n_heads = 8
            self.n_layers = 1  # Just one layer for debugging
            self.vocab_size = vocab_size
            self.max_seq_len = seq_len
            self.cond_dim = 128
            self.scale_by_sigma = False
            
    config = TestConfig()
    
    # Initialize model
    model = DiscDiffModel(config)
    model.eval()
    
    # Create test data
    indices = torch.randint(0, vocab_size-1, (batch_size, seq_len))
    sigma = torch.rand(batch_size) * 0.5
    
    print(f"Input indices shape: {indices.shape}")
    print(f"Input sigma shape: {sigma.shape}")
    
    # Test just the embedding
    x = model.vocab_embed(indices)
    print(f"After embedding: {x.shape}")
    
    # Test sigma mapping
    c = model.sigma_map(sigma)
    print(f"After sigma mapping: {c.shape}")
    
    # Test rotary embedding
    rotary_cos_sin = model.rotary_emb(x)
    print(f"Rotary cos shape: {rotary_cos_sin[0].shape}")
    print(f"Rotary sin shape: {rotary_cos_sin[1].shape}")
    
    # Test first transformer block step by step
    block = model.blocks[0]
    
    # Test modulation computation
    modulation_output = block.adaLN_modulation(c)
    print(f"Raw modulation output shape: {modulation_output.shape}")
    
    modulation_with_none = modulation_output[:, None]
    print(f"Modulation with [:, None] shape: {modulation_with_none.shape}")
    
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation_with_none.chunk(6, dim=2)
    print(f"shift_msa shape: {shift_msa.shape}")
    print(f"scale_msa shape: {scale_msa.shape}")
    
    # Test norm
    x_norm = block.norm1(x)
    print(f"After norm1: {x_norm.shape}")
    
    # Test modulation step by step
    print(f"Before modulation - x_norm: {x_norm.shape}")
    print(f"Before modulation - shift_msa: {shift_msa.shape}")
    print(f"Before modulation - scale_msa: {scale_msa.shape}")
    
    # Test the fixed modulation
    from transformer_v100 import device_compatible_modulate
    try:
        modulated = device_compatible_modulate(x_norm, shift_msa, scale_msa)
        print(f"After FIXED modulation: {modulated.shape}")
    except Exception as e:
        print(f"Error in fixed modulation: {e}")

if __name__ == "__main__":
    debug_shapes()
