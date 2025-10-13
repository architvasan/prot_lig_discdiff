#!/usr/bin/env python3
"""
Debug script to help identify hanging issues in the training script.
This script tests various components that commonly cause hangs.
"""

import os
import sys
import time
import signal
import multiprocessing as mp
from pathlib import Path

def test_with_timeout(test_func, timeout_seconds=30, test_name="Test"):
    """Run a test function with a timeout."""
    print(f"🧪 Running {test_name} (timeout: {timeout_seconds}s)...")
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"{test_name} timed out after {timeout_seconds} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        start_time = time.time()
        result = test_func()
        elapsed = time.time() - start_time
        signal.alarm(0)
        print(f"✅ {test_name} completed in {elapsed:.2f}s")
        return result
    except TimeoutError as e:
        signal.alarm(0)
        print(f"⏰ {test_name} TIMED OUT: {e}")
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"❌ {test_name} FAILED: {e}")
        return None

def test_basic_imports():
    """Test basic imports that might hang."""
    try:
        import torch
        import torch.distributed as dist
        from torch.utils.data import DataLoader
        import wandb
        from mpi4py import MPI
        return True
    except Exception as e:
        print(f"Import error: {e}")
        return False

def test_torch_multiprocessing():
    """Test PyTorch multiprocessing functionality."""
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy dataset
        data = torch.randn(100, 10)
        targets = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, targets)
        
        # Test with different worker counts
        for num_workers in [2, 1, 0]:
            try:
                loader = DataLoader(dataset, batch_size=10, num_workers=num_workers, timeout=10)
                
                # Try to get a few batches
                for i, (batch_data, batch_targets) in enumerate(loader):
                    if i >= 2:
                        break
                
                print(f"  ✅ DataLoader works with {num_workers} workers")
                return True
                
            except Exception as e:
                print(f"  ❌ DataLoader failed with {num_workers} workers: {e}")
                continue
        
        return False
        
    except Exception as e:
        print(f"PyTorch multiprocessing test failed: {e}")
        return False

def test_mpi_functionality():
    """Test MPI functionality."""
    try:
        from mpi4py import MPI
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        print(f"  MPI rank: {rank}, size: {size}")
        
        # Test broadcast
        if rank == 0:
            data = "test_data"
        else:
            data = None
        
        data = comm.bcast(data, root=0)
        print(f"  Broadcast test: {data}")
        
        return True
        
    except Exception as e:
        print(f"MPI test failed: {e}")
        return False

def test_distributed_init():
    """Test PyTorch distributed initialization."""
    try:
        import torch
        import torch.distributed as dist
        import socket
        
        # Set up environment variables
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        
        # Try to initialize
        dist.init_process_group(backend='gloo', init_method='env://', timeout=10)
        
        print(f"  Distributed initialized: rank {dist.get_rank()}, world_size {dist.get_world_size()}")
        
        # Cleanup
        dist.destroy_process_group()
        
        return True
        
    except Exception as e:
        print(f"Distributed init test failed: {e}")
        return False

def test_wandb_init():
    """Test Wandb initialization."""
    try:
        import wandb
        
        # Try to initialize wandb in offline mode
        run = wandb.init(
            project="test-project",
            name="test-run",
            mode="offline",
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True)
        )
        
        # Log a simple metric
        run.log({"test_metric": 1.0})
        
        # Finish
        run.finish()
        
        return True
        
    except Exception as e:
        print(f"Wandb test failed: {e}")
        return False

def test_file_operations():
    """Test file operations that might hang."""
    try:
        # Test creating and reading files
        test_file = "test_hang_debug.tmp"
        
        # Write test
        with open(test_file, 'w') as f:
            f.write("test data\n" * 1000)
        
        # Read test
        with open(test_file, 'r') as f:
            lines = f.readlines()
        
        # Cleanup
        os.remove(test_file)
        
        print(f"  File operations: wrote/read {len(lines)} lines")
        return True
        
    except Exception as e:
        print(f"File operations test failed: {e}")
        return False

def test_data_loading():
    """Test data loading that might hang."""
    try:
        # Try to import and test the actual data loading code
        sys.path.insert(0, str(Path(__file__).parent))
        from protlig_ddiff.utils.data_utils import UniRef50Dataset
        
        # Create a small test dataset file
        test_data_file = "test_data.pt"
        import torch
        test_data = [torch.randint(0, 25, (50,)) for _ in range(10)]
        torch.save(test_data, test_data_file)
        
        # Test dataset loading
        dataset = UniRef50Dataset(test_data_file, tokenize_on_fly=False, max_length=50)
        
        # Test getting items
        for i in range(min(3, len(dataset))):
            item = dataset[i]
            print(f"  Dataset item {i}: shape {item.shape}")
        
        # Cleanup
        os.remove(test_data_file)
        
        return True
        
    except Exception as e:
        print(f"Data loading test failed: {e}")
        return False

def main():
    """Main debug function."""
    print("🔍 Debugging potential hang issues")
    print("=" * 60)
    
    # Show environment
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"TMPDIR: {os.environ.get('TMPDIR', 'Not set')}")
    print(f"Multiprocessing start method: {mp.get_start_method()}")
    print()
    
    # Run tests
    tests = [
        (test_basic_imports, 30, "Basic imports"),
        (test_file_operations, 30, "File operations"),
        (test_torch_multiprocessing, 60, "PyTorch multiprocessing"),
        (test_mpi_functionality, 30, "MPI functionality"),
        (test_distributed_init, 30, "Distributed initialization"),
        (test_wandb_init, 60, "Wandb initialization"),
        (test_data_loading, 60, "Data loading"),
    ]
    
    results = {}
    for test_func, timeout, name in tests:
        result = test_with_timeout(test_func, timeout, name)
        results[name] = result is not None and result
        print()
    
    # Summary
    print("=" * 60)
    print("🏁 Test Summary:")
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {name}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if not results.get("PyTorch multiprocessing", False):
        print("  - Use num_workers=0 in DataLoader")
        print("  - Set TMPDIR to a shorter path")
    
    if not results.get("MPI functionality", False):
        print("  - Check MPI installation and configuration")
    
    if not results.get("Distributed initialization", False):
        print("  - Check network configuration")
        print("  - Verify MASTER_ADDR and MASTER_PORT settings")
    
    if not results.get("Wandb initialization", False):
        print("  - Use --no_wandb flag to disable wandb")
        print("  - Check network connectivity")
    
    failed_count = sum(1 for success in results.values() if not success)
    if failed_count == 0:
        print("\n🎉 All tests passed! The hanging issue might be elsewhere.")
        return 0
    else:
        print(f"\n⚠️  {failed_count} tests failed. Address these issues first.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
