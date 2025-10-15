#!/usr/bin/env python3
"""
Simple test script to verify DDP setup on Aurora.
Run with: mpiexec -n 12 -ppn 12 python test_ddp_aurora.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_mpi():
    """Test MPI availability and basic functionality."""
    print("Testing MPI...")
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print(f"MPI: Rank {rank} of {size}")
        return rank, size
    except ImportError as e:
        print(f"MPI not available: {e}")
        return None, None

def test_intel_extensions():
    """Test Intel extensions availability."""
    print("Testing Intel extensions...")
    try:
        import intel_extension_for_pytorch as ipex
        print("Intel Extension for PyTorch: Available")
    except ImportError as e:
        print(f"Intel Extension for PyTorch: Not available - {e}")
    
    try:
        import oneccl_bindings_for_pytorch as torch_ccl
        print("OneCCL bindings: Available")
    except ImportError as e:
        print(f"OneCCL bindings: Not available - {e}")

def test_torch():
    """Test PyTorch and device availability."""
    print("Testing PyTorch...")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    # Test CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
    
    # Test XPU
    try:
        print(f"XPU available: {torch.xpu.is_available()}")
        if torch.xpu.is_available():
            print(f"XPU devices: {torch.xpu.device_count()}")
    except AttributeError:
        print("XPU not available (torch.xpu not found)")
    except Exception as e:
        print(f"XPU error: {e}")

def test_ddp_setup():
    """Test DDP setup."""
    print("Testing DDP setup...")
    try:
        from protlig_ddiff.utils.ddp_utils import setup_ddp_aurora
        rank, device, world_size = setup_ddp_aurora()
        print(f"DDP setup successful: rank={rank}, device={device}, world_size={world_size}")
        
        # Test basic distributed operations
        import torch.distributed as dist
        if dist.is_initialized():
            print("Distributed is initialized")
            print(f"Backend: {dist.get_backend()}")
            
            # Test barrier
            print("Testing barrier...")
            dist.barrier()
            print("Barrier successful")
            
            # Test all_reduce
            print("Testing all_reduce...")
            tensor = torch.tensor([rank], dtype=torch.float32)
            if device.startswith('xpu'):
                tensor = tensor.to(device)
            elif device.startswith('cuda'):
                tensor = tensor.to(device)
            
            dist.all_reduce(tensor)
            print(f"All_reduce result: {tensor.item()}")
            
        return True
    except Exception as e:
        print(f"DDP setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("="*60)
    print("Aurora DDP Test")
    print("="*60)
    
    # Test MPI
    rank, size = test_mpi()
    if rank is None:
        print("❌ MPI test failed")
        return 1
    
    print(f"\n🔍 Running on rank {rank} of {size}")
    
    # Test Intel extensions
    print("\n" + "-"*40)
    test_intel_extensions()
    
    # Test PyTorch
    print("\n" + "-"*40)
    test_torch()
    
    # Test DDP setup
    print("\n" + "-"*40)
    success = test_ddp_setup()
    
    if success:
        print(f"\n✅ All tests passed on rank {rank}")
        return 0
    else:
        print(f"\n❌ Tests failed on rank {rank}")
        return 1

if __name__ == "__main__":
    exit(main())
