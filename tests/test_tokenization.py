#!/usr/bin/env python3
"""
Test tokenization to see if that's causing the hang.
"""

import os
import sys
import time
import signal
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def test_tokenizer_import():
    """Test importing the tokenizer."""
    print("🔍 Testing tokenizer import...")
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        from protlig_ddiff.utils.data_utils import ProteinTokenizer
        signal.alarm(0)
        
        print("✅ Tokenizer imported successfully")
        return True
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ Tokenizer import timed out")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Tokenizer import failed: {e}")
        return False

def test_tokenizer_creation():
    """Test creating a tokenizer instance."""
    print("\n🔍 Testing tokenizer creation...")
    
    try:
        from protlig_ddiff.utils.data_utils import ProteinTokenizer
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        tokenizer = ProteinTokenizer()
        signal.alarm(0)
        
        print(f"✅ Tokenizer created: vocab_size={tokenizer.vocab_size}")
        return tokenizer
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ Tokenizer creation timed out")
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Tokenizer creation failed: {e}")
        return None

def test_single_tokenization(tokenizer):
    """Test tokenizing a single sequence."""
    print("\n🔍 Testing single sequence tokenization...")
    
    if tokenizer is None:
        print("❌ No tokenizer to test")
        return False
    
    try:
        test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)  # 15 second timeout
        
        tokens = tokenizer.encode(test_sequence, max_length=128)
        signal.alarm(0)
        
        print(f"✅ Sequence tokenized: {tokens.shape}")
        return True
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ Single tokenization timed out")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Single tokenization failed: {e}")
        return False

def test_data_file_peek():
    """Test peeking at the actual data file format."""
    print("\n🔍 Testing data file format...")
    
    data_file = "input_data/processed_uniref50.pt"
    
    try:
        import torch
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout for just peeking
        
        # Try to load just a small portion
        data = torch.load(data_file, map_location='cpu')
        signal.alarm(0)
        
        print(f"Data type: {type(data)}")
        print(f"Data length: {len(data) if hasattr(data, '__len__') else 'unknown'}")
        
        if hasattr(data, '__getitem__'):
            # Look at first item
            item = data[0]
            print(f"First item type: {type(item)}")
            
            if isinstance(item, dict):
                print(f"First item keys: {list(item.keys())}")
                for key, value in item.items():
                    if isinstance(value, str) and len(value) > 50:
                        print(f"  {key}: {value[:50]}... (truncated)")
                    else:
                        print(f"  {key}: {value}")
            elif isinstance(item, str):
                print(f"First item (string): {item[:100]}...")
            else:
                print(f"First item: {item}")
        
        return True
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ Data file peek timed out")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Data file peek failed: {e}")
        return False

def test_dataset_with_tokenization():
    """Test dataset creation with tokenization enabled."""
    print("\n🔍 Testing dataset with tokenization...")
    
    try:
        from protlig_ddiff.utils.data_utils import UniRef50Dataset
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 60 second timeout
        
        # Create dataset with tokenization
        dataset = UniRef50Dataset(
            data_file="input_data/processed_uniref50.pt",
            tokenize_on_fly=True,  # Enable tokenization
            max_length=128,        # Shorter for testing
            use_streaming=True     # Use streaming to avoid loading all into memory
        )
        signal.alarm(0)
        
        print(f"✅ Dataset with tokenization created: {len(dataset)} samples")
        
        # Test getting one item
        signal.alarm(30)
        item = dataset[0]
        signal.alarm(0)
        
        print(f"✅ First item tokenized: {item.shape}")
        return True
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ Dataset with tokenization timed out")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Dataset with tokenization failed: {e}")
        return False

def main():
    print("🔍 TESTING TOKENIZATION (LIKELY HANG CAUSE)")
    print("=" * 60)
    
    # Test each component
    if not test_tokenizer_import():
        print("\n❌ Tokenizer import failed")
        return
    
    tokenizer = test_tokenizer_creation()
    if tokenizer is None:
        print("\n❌ Tokenizer creation failed")
        return
    
    if not test_single_tokenization(tokenizer):
        print("\n❌ Single tokenization failed")
        return
    
    if not test_data_file_peek():
        print("\n❌ Data file peek failed")
        return
    
    if not test_dataset_with_tokenization():
        print("\n❌ Dataset with tokenization failed - THIS IS LIKELY YOUR HANG!")
        return
    
    print("\n✅ All tokenization tests passed!")
    print("\n💡 If tests pass but training still hangs, the issue is likely:")
    print("1. Batch size too large for tokenization")
    print("2. Model initialization hanging")
    print("3. GPU memory issues")

if __name__ == "__main__":
    main()
