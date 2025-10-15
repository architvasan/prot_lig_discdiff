"""
Distributed Data Parallel (DDP) utilities for Aurora and Polaris clusters.
"""
# Fix for AF_UNIX path too long error on HPC systems
def setup_temp_directory_if_needed():
    """Setup a shorter temporary directory to avoid AF_UNIX path too long errors."""
    # Only setup if TMPDIR is not already set to a short path
    current_tmpdir = os.environ.get('TMPDIR', '/tmp')
    if len(current_tmpdir) > 80:  # Arbitrary threshold for "too long"
        short_tmp_dirs = ['/tmp', '/dev/shm', '.']

        for tmp_dir in short_tmp_dirs:
            if os.path.exists(tmp_dir) and os.access(tmp_dir, os.W_OK):
                import tempfile
                try:
                    temp_dir = tempfile.mkdtemp(prefix='ddp_', dir=tmp_dir)
                    os.environ['TMPDIR'] = temp_dir
                    os.environ['TEMP'] = temp_dir
                    os.environ['TMP'] = temp_dir
                    # print(f"🔧 DDP: Set temporary directory to: {temp_dir}")
                    return temp_dir
                except Exception as e:
                    # print(f"⚠️  DDP: Failed to create temp dir in {tmp_dir}: {e}")
                    continue

# Optional MPI import for Aurora
#from mpi4py import MPI
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    # print("⚠️  MPI not available - DDP will be disabled")
import os
import socket
import torch
import torch.distributed as dist

# Intel extensions for Aurora
#try:
#    import intel_extension_for_pytorch as ipex
#    import oneccl_bindings_for_pytorch as torch_ccl
#    INTEL_AVAILABLE = True
#except ImportError:
#    INTEL_AVAILABLE = False
#    print("⚠️  Intel extensions not found, running on CPU/CUDA")


def setup_ddp_aurora():
    """Setup DDP for Aurora with proper Intel XPU handling."""
    if not MPI_AVAILABLE:
        raise RuntimeError("MPI not available - cannot setup Aurora DDP")

    # Setup temp directory to avoid AF_UNIX path too long errors
    setup_temp_directory_if_needed()

    # DDP: Set environmental variables used by PyTorch
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()
    LOCAL_RANK = int(os.environ.get('PALS_LOCAL_RANKID', '0'))

    # Set environment variables
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    os.environ['LOCAL_RANK'] = str(LOCAL_RANK)

    # Setup master address for Aurora
    MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = f"{MASTER_ADDR}.hsn.cm.aurora.alcf.anl.gov"
    os.environ['MASTER_PORT'] = str(2345)

    print(f"DDP: Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK}. {MASTER_ADDR}")

    ## Set XPU device
    #device = f'xpu:{LOCAL_RANK}'
    #print(device)
    #torch.xpu.set_device(LOCAL_RANK)

    # Initialize distributed communication with CCL backend for Intel XPU
    torch.distributed.init_process_group(
        backend='xpu:ccl',
        init_method='env://',
        rank=int(RANK),
        world_size=int(SIZE)
    )

    # Set CUDA device
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    
    return dist.get_rank(), device_id, dist.get_world_size()



    # print(f"✅ DDP initialized: rank {RANK}/{SIZE}, local_rank {LOCAL_RANK}, device {device}")

    #return RANK, device, SIZE


def setup_ddp_polaris(rank, world_size):
    """Setup DDP for Polaris cluster."""
    if not MPI_AVAILABLE:
        raise RuntimeError("MPI not available - cannot setup Polaris DDP")

    # Setup temp directory to avoid AF_UNIX path too long errors
    setup_temp_directory_if_needed()

    # DDP: Set environmental variables used by PyTorch
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()
    LOCAL_RANK = int(os.environ.get('PMI_LOCAL_RANK', '0'))

    # Set environment variables
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    os.environ['LOCAL_RANK'] = str(LOCAL_RANK)

    # Setup master address for Polaris
    MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = str(12345)

    print(f"DDP: Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK}. {MASTER_ADDR}")

    # Initialize distributed communication
    torch.distributed.init_process_group(backend='nccl', init_method='env://', rank = int(RANK), world_size = int(SIZE))

    # Set CUDA device
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    
    return dist.get_rank(), device_id, dist.get_world_size()


def cleanup_ddp():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        # print("✅ DDP cleanup completed")


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size():
    """Get the world size for distributed training."""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank():
    """Get the current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor):
    """All-reduce a tensor and compute the mean across all processes."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor
