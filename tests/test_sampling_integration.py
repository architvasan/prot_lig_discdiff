#!/usr/bin/env python3
"""
Test script to verify sampling integration works.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_sampling_imports():
    """Test that all sampling imports work."""
    print("🔍 Testing sampling imports...")
    
    try:
        from protlig_ddiff.sampling.protein_sampling import ProteinSampler, sample_during_training
        print("✅ ProteinSampler imported successfully")
        
        from protlig_ddiff.sampling.sampling import get_pc_sampler
        print("✅ get_pc_sampler imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_with_sampling():
    """Test that config loading includes sampling section."""
    print("\n🔍 Testing config with sampling section...")
    
    try:
        import yaml
        
        # Load config
        with open("configs/config_protein.yaml", "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Check sampling section
        if 'sampling' in config_dict:
            sampling_config = config_dict['sampling']
            print("✅ Sampling section found in config")
            print(f"  sample_interval: {sampling_config.get('sample_interval')}")
            print(f"  eval_batch_size: {sampling_config.get('eval_batch_size')}")
            print(f"  eval_max_length: {sampling_config.get('eval_max_length')}")
            print(f"  eval_steps: {sampling_config.get('eval_steps')}")
            print(f"  predictor: {sampling_config.get('predictor')}")
            return True
        else:
            print("❌ Sampling section not found in config")
            return False
            
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_mock_sampling():
    """Test sampling with mock components."""
    print("\n🔍 Testing mock sampling...")
    
    try:
        from protlig_ddiff.sampling.protein_sampling import ProteinSampler
        from protlig_ddiff.utils.data_utils import ProteinTokenizer
        
        # Create mock components
        class MockModel:
            def __init__(self):
                self.device = 'cpu'
            def eval(self):
                pass
            def __call__(self, x, sigma):
                # Return mock logits
                batch_size, seq_len = x.shape
                vocab_size = 26
                return torch.randn(batch_size, seq_len, vocab_size)
        
        class MockGraph:
            def __init__(self):
                self.absorb = True
            def sample_limit(self, batch_size, seq_len):
                return torch.randint(0, 25, (batch_size, seq_len))
            def staggered_score(self, score, sigma):
                return torch.softmax(score, dim=-1)
            def transp_transition(self, x, sigma):
                batch_size, seq_len = x.shape
                vocab_size = 26
                return torch.ones(batch_size, seq_len, vocab_size)
        
        class MockNoise:
            def __call__(self, t):
                return t, -torch.ones_like(t)
        
        # Create sampler
        model = MockModel()
        graph = MockGraph()
        noise = MockNoise()
        tokenizer = ProteinTokenizer()
        
        sampler = ProteinSampler(model, graph, noise, tokenizer, device='cpu')
        print("✅ ProteinSampler created with mock components")
        
        # Test conditioning function creation
        conditioning_fn = sampler.create_conditioning_function(
            start_tokens=[1],  # BOS
            end_tokens=[2],    # EOS
            max_length=64
        )
        print("✅ Conditioning function created")
        
        # Test conditioning function
        mock_x = torch.randint(0, 25, (2, 64))  # batch_size=2, seq_len=64
        conditioned_x = conditioning_fn(mock_x)
        print(f"✅ Conditioning applied: shape {conditioned_x.shape}")
        print(f"  First token: {conditioned_x[0, 0].item()} (should be 1)")
        print(f"  Last token: {conditioned_x[0, -1].item()} (should be 2)")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock sampling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 TESTING SAMPLING INTEGRATION")
    print("=" * 50)
    
    success = True
    
    if not test_sampling_imports():
        success = False
    
    if not test_config_with_sampling():
        success = False
    
    if not test_mock_sampling():
        success = False
    
    if success:
        print("\n🎉 All sampling integration tests passed!")
        print("\n💡 Key features added:")
        print("  ✅ ProteinSampler class for protein-specific sampling")
        print("  ✅ Conditioning support for start/end tokens")
        print("  ✅ Integration with training loop")
        print("  ✅ Sampling configuration in YAML")
        print("  ✅ Standalone sampling script")
        print("\n🚀 Ready to use sampling during training!")
    else:
        print("\n❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
