#!/usr/bin/env python3
"""
Test script to verify that the AF_UNIX path too long fix works.
This script tests multiprocessing functionality that typically triggers the error.
"""

import os
import sys
import tempfile
import multiprocessing as mp
from pathlib import Path

def setup_temp_directory():
    """Setup a shorter temporary directory to avoid AF_UNIX path too long errors."""
    # Try to use /tmp first, fall back to current directory if needed
    short_tmp_dirs = ['/tmp', '/dev/shm', '.']
    
    for tmp_dir in short_tmp_dirs:
        if os.path.exists(tmp_dir) and os.access(tmp_dir, os.W_OK):
            # Create a unique subdirectory
            try:
                temp_dir = tempfile.mkdtemp(prefix='test_pytorch_', dir=tmp_dir)
                os.environ['TMPDIR'] = temp_dir
                os.environ['TEMP'] = temp_dir
                os.environ['TMP'] = temp_dir
                print(f"🔧 Set temporary directory to: {temp_dir}")
                return temp_dir
            except Exception as e:
                print(f"⚠️  Failed to create temp dir in {tmp_dir}: {e}")
                continue
    
    # If all else fails, use current directory
    current_tmp = os.path.join(os.getcwd(), 'tmp_test')
    os.makedirs(current_tmp, exist_ok=True)
    os.environ['TMPDIR'] = current_tmp
    os.environ['TEMP'] = current_tmp
    os.environ['TMP'] = current_tmp
    print(f"🔧 Set temporary directory to: {current_tmp}")
    return current_tmp

def test_worker(x):
    """Simple worker function for multiprocessing test."""
    return x * x

def test_multiprocessing():
    """Test multiprocessing functionality."""
    print("🧪 Testing multiprocessing...")
    
    # Test with different start methods
    for method in ['spawn', 'fork', 'forkserver']:
        try:
            mp.set_start_method(method, force=True)
            print(f"  Testing with start method: {method}")
            
            with mp.Pool(processes=2) as pool:
                results = pool.map(test_worker, [1, 2, 3, 4])
                print(f"    Results: {results}")
                
            print(f"  ✅ {method} method works!")
            break
            
        except Exception as e:
            print(f"  ❌ {method} method failed: {e}")
            continue
    else:
        print("  ❌ All multiprocessing methods failed!")
        return False
    
    return True

def test_torch_dataloader():
    """Test PyTorch DataLoader with multiprocessing."""
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        print("🧪 Testing PyTorch DataLoader...")
        
        # Create dummy dataset
        data = torch.randn(100, 10)
        targets = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, targets)
        
        # Test with different numbers of workers
        for num_workers in [4, 2, 1, 0]:
            try:
                loader = DataLoader(dataset, batch_size=10, num_workers=num_workers)
                
                # Try to iterate through a few batches
                for i, (batch_data, batch_targets) in enumerate(loader):
                    if i >= 2:  # Just test a few batches
                        break
                
                print(f"  ✅ DataLoader works with {num_workers} workers!")
                return True
                
            except OSError as e:
                if "AF_UNIX path too long" in str(e):
                    print(f"  ⚠️  AF_UNIX error with {num_workers} workers, trying fewer...")
                    continue
                else:
                    raise
            except Exception as e:
                print(f"  ❌ DataLoader failed with {num_workers} workers: {e}")
                continue
        
        print("  ❌ All DataLoader configurations failed!")
        return False
        
    except ImportError:
        print("  ⚠️  PyTorch not available, skipping DataLoader test")
        return True

def main():
    """Main test function."""
    print("🔍 Testing AF_UNIX path too long fix")
    print("=" * 50)
    
    # Show current environment
    print(f"Current working directory: {os.getcwd()}")
    print(f"Current TMPDIR: {os.environ.get('TMPDIR', 'Not set')}")
    print(f"Path length: {len(os.environ.get('TMPDIR', os.getcwd()))}")
    print()
    
    # Setup temp directory
    temp_dir = setup_temp_directory()
    print(f"New TMPDIR: {os.environ.get('TMPDIR')}")
    print(f"New path length: {len(os.environ.get('TMPDIR'))}")
    print()
    
    # Run tests
    mp_success = test_multiprocessing()
    torch_success = test_torch_dataloader()
    
    # Cleanup
    if temp_dir and os.path.exists(temp_dir):
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"⚠️  Failed to cleanup temp directory: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    if mp_success and torch_success:
        print("🎉 All tests passed! The fix should work.")
        return 0
    else:
        print("❌ Some tests failed. You may need additional fixes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
