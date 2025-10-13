#!/usr/bin/env python3
"""
Debug script to identify exactly where the first batch hang occurs.
"""

import os
import sys
import time
import torch
import signal
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def test_data_file():
    """Test if the data file can be opened and read."""
    print("🔍 Testing data file access...")
    data_file = "input_data/processed_uniref50.pt"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return False
    
    try:
        # Test file size
        size_mb = os.path.getsize(data_file) / (1024 * 1024)
        print(f"📊 Data file size: {size_mb:.1f} MB")
        
        # Test if it's a torch file
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout
        
        data = torch.load(data_file, map_location='cpu')
        signal.alarm(0)
        
        print(f"✅ Data file loaded: {type(data)}, length: {len(data) if hasattr(data, '__len__') else 'unknown'}")
        
        # Test first few items
        if hasattr(data, '__getitem__'):
            for i in range(min(3, len(data))):
                item = data[i]
                print(f"  Item {i}: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'no shape'}")
        
        return True
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ Data file loading timed out")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Error loading data file: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation without DataLoader."""
    print("\n🔍 Testing dataset creation...")
    
    try:
        from protlig_ddiff.utils.data_utils import UniRef50Dataset
        
        # Test with minimal settings
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        dataset = UniRef50Dataset(
            data_file="input_data/processed_uniref50.pt",
            tokenize_on_fly=False,
            max_length=512,
            use_streaming=False  # Try memory loading first
        )
        signal.alarm(0)
        
        print(f"✅ Dataset created: {len(dataset)} samples")
        return dataset
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ Dataset creation timed out")
        
        # Try streaming mode
        try:
            print("🔄 Trying streaming mode...")
            signal.alarm(30)
            dataset = UniRef50Dataset(
                data_file="input_data/processed_uniref50.pt",
                tokenize_on_fly=False,
                max_length=512,
                use_streaming=True
            )
            signal.alarm(0)
            print(f"✅ Streaming dataset created: {len(dataset)} samples")
            return dataset
        except Exception as e:
            signal.alarm(0)
            print(f"❌ Streaming dataset creation failed: {e}")
            return None
            
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Dataset creation failed: {e}")
        return None

def test_single_item(dataset):
    """Test getting a single item from dataset."""
    print("\n🔍 Testing single item access...")
    
    if dataset is None:
        print("❌ No dataset to test")
        return None
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)  # 15 second timeout
        
        item = dataset[0]
        signal.alarm(0)
        
        print(f"✅ Single item loaded: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'no shape'}")
        return item
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ Single item access timed out")
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Single item access failed: {e}")
        return None

def test_dataloader_creation(dataset):
    """Test DataLoader creation."""
    print("\n🔍 Testing DataLoader creation...")
    
    if dataset is None:
        print("❌ No dataset to test")
        return None
    
    try:
        from torch.utils.data import DataLoader
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,  # Very small batch
            shuffle=False,
            num_workers=0,  # No multiprocessing
            pin_memory=False,
            drop_last=True
        )
        signal.alarm(0)
        
        print("✅ DataLoader created successfully")
        return dataloader
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ DataLoader creation timed out")
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"❌ DataLoader creation failed: {e}")
        return None

def test_first_batch(dataloader):
    """Test getting the first batch."""
    print("\n🔍 Testing first batch...")
    
    if dataloader is None:
        print("❌ No dataloader to test")
        return None
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        print("🔄 Creating iterator...")
        iterator = iter(dataloader)
        
        print("🔄 Getting first batch...")
        batch = next(iterator)
        signal.alarm(0)
        
        print(f"✅ First batch loaded: {type(batch)}, shape: {batch.shape if hasattr(batch, 'shape') else 'no shape'}")
        return batch
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ First batch loading timed out")
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"❌ First batch loading failed: {e}")
        return None

def test_device_transfer(batch):
    """Test transferring batch to GPU."""
    print("\n🔍 Testing device transfer...")
    
    if batch is None:
        print("❌ No batch to test")
        return False
    
    try:
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping GPU test")
            return True
        
        device = torch.device('cuda:0')
        print(f"🔄 Transferring to {device}...")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout
        
        batch_gpu = batch.to(device)
        signal.alarm(0)
        
        print(f"✅ Batch transferred to GPU: {batch_gpu.device}")
        return True
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ Device transfer timed out")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Device transfer failed: {e}")
        return False

def main():
    print("🔍 DEBUGGING FIRST BATCH HANG")
    print("=" * 50)
    
    # Test each component step by step
    if not test_data_file():
        print("\n❌ Data file test failed - check your data file")
        return
    
    dataset = test_dataset_creation()
    if dataset is None:
        print("\n❌ Dataset creation failed - check data format")
        return
    
    item = test_single_item(dataset)
    if item is None:
        print("\n❌ Single item access failed - check dataset implementation")
        return
    
    dataloader = test_dataloader_creation(dataset)
    if dataloader is None:
        print("\n❌ DataLoader creation failed")
        return
    
    batch = test_first_batch(dataloader)
    if batch is None:
        print("\n❌ First batch loading failed - this is likely where your hang occurs")
        return
    
    if not test_device_transfer(batch):
        print("\n❌ Device transfer failed")
        return
    
    print("\n✅ All tests passed! The issue might be elsewhere.")
    print("\n💡 Suggestions:")
    print("1. Check if your model initialization is hanging")
    print("2. Check if wandb initialization is hanging")
    print("3. Try running with --no_wandb")
    print("4. Check your config file for any issues")

if __name__ == "__main__":
    main()
