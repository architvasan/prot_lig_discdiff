"""
Clean and organized training script for UniRef50 discrete diffusion training.
"""
from mpi4py import MPI
# print("loaded MPI")
import os
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import yaml

# Fix for AF_UNIX path too long error on HPC systems
# Set a shorter temporary directory path before any multiprocessing operations
def setup_temp_directory():
    """Setup a shorter temporary directory to avoid AF_UNIX path too long errors."""
    # Try to use /tmp first, fall back to current directory if needed
    short_tmp_dirs = ['/tmp', '/dev/shm', '.']

    for tmp_dir in short_tmp_dirs:
        if os.path.exists(tmp_dir) and os.access(tmp_dir, os.W_OK):
            # Create a unique subdirectory
            import tempfile
            try:
                temp_dir = tempfile.mkdtemp(prefix='pytorch_', dir=tmp_dir)
                os.environ['TMPDIR'] = temp_dir
                os.environ['TEMP'] = temp_dir
                os.environ['TMP'] = temp_dir
                # print(f"🔧 Set temporary directory to: {temp_dir}")
                return temp_dir
            except Exception as e:
                # print(f"⚠️  Failed to create temp dir in {tmp_dir}: {e}")
                pass
                continue

    # If all else fails, use current directory
    current_tmp = os.path.join(os.getcwd(), 'tmp_pytorch')
    os.makedirs(current_tmp, exist_ok=True)
    os.environ['TMPDIR'] = current_tmp
    os.environ['TEMP'] = current_tmp
    os.environ['TMP'] = current_tmp
    # print(f"🔧 Set temporary directory to: {current_tmp}")
    return current_tmp

# Setup temp directory before any other imports that might use multiprocessing
_temp_dir_created = setup_temp_directory()

def cleanup_temp_directory():
    """Clean up the temporary directory we created."""
    global _temp_dir_created
    if _temp_dir_created and os.path.exists(_temp_dir_created):
        try:
            import shutil
            shutil.rmtree(_temp_dir_created)
            # print(f"🧹 Cleaned up temporary directory: {_temp_dir_created}")
        except Exception as e:
            # print(f"⚠️  Failed to cleanup temp directory {_temp_dir_created}: {e}")
            pass

import torch
try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch
    INTEL_GPU=True
except:
    INTEL_GPU=False

import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import numpy as np
from tqdm import tqdm
import socket

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import utilities
from protlig_ddiff.utils.ddp_utils import  cleanup_ddp, is_main_process, get_rank, get_world_size, setup_ddp_polaris, setup_ddp_aurora
from protlig_ddiff.utils.data_utils import UniRef50Dataset, ProteinTokenizer
from protlig_ddiff.utils.config_utils import load_config, create_namespace_from_dict, print_config_summary
from protlig_ddiff.utils.training_utils import (
    create_optimizer, create_scheduler, EMAModel, TrainingMetrics,
    compute_accuracy, compute_perplexity, clip_gradients,
    save_checkpoint, load_checkpoint, setup_wandb, log_metrics,
    format_time, estimate_remaining_time
)

# Import SUBS loss
from protlig_ddiff.processing.subs_loss import subs_loss, subs_loss_with_curriculum, compute_subs_metrics

# Import model and other components
import protlig_ddiff.processing.graph_lib as graph_lib
import protlig_ddiff.processing.noise_lib as noise_lib
from protlig_ddiff.models.transformer_v100 import DiscDiffModel

# Import sampling
from protlig_ddiff.sampling.protein_sampling import sample_during_training

#### Setup DDP on Polaris
####
def _setup_ddp_polaris(rank, world_size):
    """Setup DDP for Polaris cluster."""
    #from mpi4py import MPI
    MPI_AVAILABLE=True
    if not MPI_AVAILABLE:
        raise RuntimeError("MPI not available - cannot setup Polaris DDP")

    # DDP: Set environmental variables used by PyTorch
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()
    # print(SIZE, RANK)
    LOCAL_RANK = int(os.environ.get('PMI_LOCAL_RANK', '0'))

    # Set environment variables
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    os.environ['LOCAL_RANK'] = str(LOCAL_RANK)

    # Setup master address for Polaris
    MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = str(2345)

    # print(f"DDP: Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK}. {MASTER_ADDR}")

    # Initialize distributed communication
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Set CUDA device - CRITICAL FIX: use RANK not rank parameter
    device_id = RANK % torch.cuda.device_count()
    torch.cuda.set_device(device_id)

    # Verify DDP setup
    actual_rank = dist.get_rank()
    actual_world_size = dist.get_world_size()
    # print(f"🔍 DDP Verification - MPI rank: {RANK}, PyTorch rank: {actual_rank}, device: {device_id}")

    if RANK != actual_rank:
        # print(f"🚨 WARNING: MPI rank ({RANK}) != PyTorch rank ({actual_rank})")
        pass

    return actual_rank, device_id, actual_world_size

### Main Training config

@dataclass
class _TrainerConfig:
    """Configuration for the trainer."""
    work_dir: str
    config_file: str
    datafile: str
    rank: int = 0
    world_size: int = 1
    device: str = 'cpu'
    devicetype: str = 'cuda'
    seed: int = 42
    use_wandb: bool = True
    resume_checkpoint: Optional[str] = None

@dataclass
class TrainerConfig:
    """Enhanced trainer configuration that loads and includes all YAML arguments."""
    work_dir: str
    datafile: str
    config_file: str | None = None
    rank: int = 0
    world_size: int = 1
    device: str = 'xpu'
    seed: int = 42
    use_wandb: bool = True
    resume_checkpoint: Optional[str] = None

    # YAML config sections will be added dynamically
    model: Optional[dict] = None
    training: Optional[dict] = None
    data: Optional[dict] = None
    noise: Optional[dict] = None
    curriculum: Optional[dict] = None
    logging: Optional[dict] = None
    optimizer: Optional[dict] = None
    scheduler: Optional[dict] = None

    # Top-level config values
    tokens: int = 26
    devicetype: str = 'xpu'

    def __post_init__(self):
        """Load YAML config and set all values as attributes."""
        if self.config_file is not None:
            import yaml
            with open(self.config_file, "r") as f:
                config_dict = yaml.safe_load(f)

            # Set all config values as attributes
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    # Convert dict to namespace-like object for easy access
                    setattr(self, key, self._dict_to_namespace(value))
                else:
                    setattr(self, key, value)

    def _dict_to_namespace(self, d):
        """Convert dictionary to namespace-like object for dot notation access."""
        class ConfigNamespace:
            def __init__(self, dictionary):
                for key, value in dictionary.items():
                    if isinstance(value, dict):
                        setattr(self, key, ConfigNamespace(value))
                    else:
                        setattr(self, key, value)

            def __getattr__(self, name):
                # Return None for missing attributes instead of raising AttributeError
                return None

            def get(self, key, default=None):
                """Get attribute with default value."""
                return getattr(self, key, default)

        return ConfigNamespace(d)

    def __getitem__(self, key):
        """Allow dictionary-style access."""
        return getattr(self, key)

    def get(self, key, default=None):
        """Get attribute with default value."""
        return getattr(self, key, default)


class UniRef50Trainer:
    """Clean and organized trainer for UniRef50 discrete diffusion training."""
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.setup_environment()
        self.setup_validation_tracking()
        self.setup_model_and_data()
        self.setup_training_components()
        
    def setup_environment(self):
        """Setup training environment and device."""
        # Set random seeds
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Setup device with improved logic
        device_str = str(self.config.device).lower()

        if device_str == 'cpu':
            self.device = torch.device('cpu')
        elif device_str.startswith('cuda:') or device_str.startswith('xpu:'):
            # Full device specification like 'cuda:0' or 'xpu:0'
            self.device = torch.device(device_str)
        elif device_str.isdigit():
            # Just a number like '0', '1', etc. - assume CUDA
            self.device = torch.device(f'cuda:{device_str}')
        elif self.config.devicetype in ['cuda', 'xpu']:
            # Legacy devicetype + device combination
            self.device = torch.device(f'{self.config.devicetype}:{self.config.device}')
        else:
            # Default to CPU
            self.device = torch.device('cpu')

        # Verify device availability
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                # print("⚠️  CUDA not available, falling back to CPU")
                self.device = torch.device('cpu')
            elif self.device.index is not None and self.device.index >= torch.cuda.device_count():
                # print(f"⚠️  GPU {self.device.index} not available, using GPU 0")
                self.device = torch.device('cuda:0')

        # print(f"🔧 Environment setup: device={self.device}, seed={self.config.seed}")

        # Set device if using GPU or XPU
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        elif self.device.type == 'xpu':
            try:
                torch.xpu.set_device(self.device)
            except Exception as e:
                print(f"⚠️  Could not set XPU device {self.device}: {e}")

    def setup_validation_tracking(self):
        """Initialize validation tracking variables."""
        # Initialize validation tracking
        self.best_val_loss = float('inf')
        self.val_loss_history = []
        self.steps_without_improvement = 0
        self.last_checkpoint_step = 0

        # Get validation config with defaults
        val_config = getattr(self.config, 'validation', {})
        self.val_eval_freq = getattr(val_config, 'eval_freq', 500)
        self.checkpoint_freq = getattr(val_config, 'checkpoint_freq', 1000)
        self.checkpoint_on_improvement = getattr(val_config, 'checkpoint_on_improvement', True)
        self.patience = getattr(val_config, 'patience', 10)
        self.min_delta = getattr(val_config, 'min_delta', 0.001)
        self.save_best_only = getattr(val_config, 'save_best_only', False)
        self.val_batch_limit = getattr(val_config, 'val_batch_limit', 20)
                # Don't fail, just continue
        
    def setup_model_and_data(self):
        """Setup model, data, and related components."""
        # Print config summary
        if is_main_process():
            # print("📋 Configuration Summary:")
            # print(f"  Model: dim={self.config.model.dim}, layers={self.config.model.n_layers}, heads={self.config.model.n_heads}")
            # print(f"  Training: lr={self.config.training.learning_rate}, batch_size={self.config.training.batch_size}")
            # print(f"  Data: max_length={self.config.data.max_length}, tokenize_on_fly={self.config.data.tokenize_on_fly}")
            # print(f"  Noise: type={self.config.noise.type}")
            pass

        # Setup graph and noise
        vocab_size = self.config.tokens
        self.graph = graph_lib.Absorbing(vocab_size - 1)  # vocab_size includes absorbing token

        # Setup noise based on config
        noise_type = self.config.noise.type.lower()
        if noise_type == 'loglinear':
            self.noise = noise_lib.LogLinearNoise()
        elif noise_type == 'cosine':
            self.noise = noise_lib.CosineNoise()
        else:
            # print(f"⚠️  Unknown noise type: {noise_type}, defaulting to LogLinear")
            self.noise = noise_lib.LogLinearNoise()

        # Setup model - pass the config directly
        self.model = DiscDiffModel(self.config).to(self.device)
        
        # Setup DDP if needed
        if self.config.world_size > 1:
            # For CUDA, specify device_ids; for XPU/CPU, use None
            device_ids = [self.device] if self.device.type == 'cuda' else None
            self.model = DDP(self.model, device_ids=device_ids)
        
        # Setup data
        self.setup_data_loaders()
        
        # print(f"🏗️  Model setup: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        # print(f"📊 Data setup: {len(self.train_loader)} batches per epoch")
    
    def create_data_splits(self, dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, split_seed=42):
        """Create train/validation/test splits from dataset."""
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Data split ratios must sum to 1.0, got {total_ratio}")

        dataset_size = len(dataset)

        # Calculate split sizes
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size  # Remainder goes to test

        print(f"📊 Data splits: Train={train_size:,} ({train_ratio:.1%}), "
              f"Val={val_size:,} ({val_ratio:.1%}), Test={test_size:,} ({test_ratio:.1%})")

        # Create reproducible random split
        generator = torch.Generator().manual_seed(split_seed)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )

        return train_dataset, val_dataset, test_dataset

    def setup_data_loaders(self):
        """Setup training, validation, and test data loaders with proper splits."""
        # Set multiprocessing start method to avoid issues on HPC systems
        import multiprocessing as mp
        try:
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)
                # print("🔧 Set multiprocessing start method to 'spawn'")
                pass
        except RuntimeError as e:
            # print(f"⚠️  Could not set multiprocessing start method: {e}")
            pass

        # Add timeout for data loading operations
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Data loading operation timed out")

        # Training dataset with timeout protection
        # print("📂 Loading training dataset...")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 minute timeout for dataset loading

        try:
            # Get data settings from config
            data_config = getattr(self.config, 'data', {})
            tokenize_on_fly = getattr(data_config, 'tokenize_on_fly', True)
            use_streaming = getattr(data_config, 'use_streaming', True)
            max_length = getattr(data_config, 'max_length', 256)

            # Disable streaming for .pt files (binary format doesn't support streaming)
            if self.config.datafile.endswith('.pt'):
                use_streaming = False
                # print("🔧 Disabled streaming for .pt file (binary format)")

            # print(f"🔧 Dataset settings: tokenize_on_fly={tokenize_on_fly}, use_streaming={use_streaming}, max_length={max_length}")

            # Load full dataset first
            full_dataset = UniRef50Dataset(
                data_file=self.config.datafile,
                tokenize_on_fly=tokenize_on_fly,
                max_length=max_length,
                use_streaming=use_streaming
            )
            signal.alarm(0)  # Cancel timeout
            # print(f"✅ Full dataset loaded successfully: {len(full_dataset)} samples")

            # Create train/val/test splits
            train_ratio = getattr(data_config, 'train_ratio', 0.8)
            val_ratio = getattr(data_config, 'val_ratio', 0.1)
            test_ratio = getattr(data_config, 'test_ratio', 0.1)
            split_seed = getattr(data_config, 'split_seed', 42)

            train_dataset, val_dataset, test_dataset = self.create_data_splits(
                full_dataset, train_ratio, val_ratio, test_ratio, split_seed
            )

        except TimeoutError:
            signal.alarm(0)
            # print("⚠️  Dataset loading timed out, trying with streaming=True")
            full_dataset = UniRef50Dataset(
                data_file=self.config.datafile,
                tokenize_on_fly=getattr(data_config, 'tokenize_on_fly', False),
                max_length=getattr(data_config, 'max_length', 512),
                use_streaming=True  # Force streaming on timeout
            )

            # Create splits even with timeout fallback
            data_config = getattr(self.config, 'data', {})
            train_ratio = getattr(data_config, 'train_ratio', 0.8)
            val_ratio = getattr(data_config, 'val_ratio', 0.1)
            test_ratio = getattr(data_config, 'test_ratio', 0.1)
            split_seed = getattr(data_config, 'split_seed', 42)

            train_dataset, val_dataset, test_dataset = self.create_data_splits(
                full_dataset, train_ratio, val_ratio, test_ratio, split_seed
            )
        
        # Setup sampler for DDP
        train_sampler = None
        if self.config.world_size > 1:
            # print(f"🔍 DDP Debug - Rank {self.config.rank}/{self.config.world_size}: Setting up DistributedSampler")
            # print(f"   Dataset size: {len(train_dataset)}")

            # Use rank-specific seed to ensure different sampling per rank
            sampler_seed = self.config.seed + self.config.rank * 1000
            # print(f"   Using sampler seed: {sampler_seed} (base: {self.config.seed}, rank: {self.config.rank})")

            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=True,
                seed=sampler_seed  # Rank-specific seed
            )

            # print(f"   Sampler created - rank {self.config.rank} will see {len(train_sampler)} samples")

            # Test the sampler by getting first few indices
            sampler_indices = list(train_sampler)[:5]
            # print(f"   First 5 sampler indices for rank {self.config.rank}: {sampler_indices}")

            # CRITICAL TEST: Check if dataset returns different items for different indices
            # print(f"🔍 Dataset test for rank {self.config.rank}:")
            for i, idx in enumerate(sampler_indices[:3]):
                item = train_dataset[idx]
                item_hash = hash(tuple(item[:20].numpy())) if hasattr(item, 'numpy') else hash(tuple(item[:20]))
                # print(f"   Index {idx} → hash: {item_hash}, first 10: {item[:10]}")
                if i > 0 and item_hash == prev_hash:
                    # print(f"🚨 DATASET BUG: Index {idx} returns same data as previous index!")
                    pass
                prev_hash = item_hash
        else:
            # print(f"🔍 Single process training - no DistributedSampler needed")
            pass
        
        # Training data loader with fallback for num_workers
        data_config = getattr(self.config, 'data', {})
        num_workers = getattr(data_config, 'num_workers', 4)

        # For HPC systems, start with fewer workers to avoid hangs
        if self.config.world_size > 1:
            num_workers = min(num_workers, 2)  # Limit workers in distributed mode
            # print(f"🔧 Limited num_workers to {num_workers} for distributed training")

        # For large datasets, start with fewer workers to avoid memory issues
        dataset_size = len(train_dataset)
        if dataset_size > 1000000:  # 1M+ sequences
            num_workers = min(num_workers, 1)  # Start with 1 worker for large datasets
            # print(f"🔧 Large dataset detected ({dataset_size:,} samples), starting with {num_workers} worker(s)")

        # Try to create data loader, reduce num_workers if issues occur
        training_config = getattr(self.config, 'training', {})
        batch_size = getattr(training_config, 'batch_size', 32)
        pin_memory = getattr(data_config, 'pin_memory', True)

        for workers in [num_workers, max(1, num_workers // 2), 1, 0]:
            try:
                # print(f"🔧 Trying DataLoader with {workers} workers...")
                # Debug: Print DataLoader configuration
                shuffle_setting = (train_sampler is None)
                # print(f"🔍 DataLoader config - shuffle: {shuffle_setting}, sampler: {train_sampler is not None}")

                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=shuffle_setting,  # Only shuffle if no sampler
                    sampler=train_sampler,
                    num_workers=workers,
                    pin_memory=pin_memory and workers > 0,
                    drop_last=True,
                    timeout=30 if workers > 0 else 0,  # Add timeout for worker processes
                    persistent_workers=False,  # Disable persistent workers to avoid hangs
                    generator=torch.Generator().manual_seed(self.config.seed + self.config.rank)  # Rank-specific seed
                )

                # Test the data loader by getting one batch
                if workers == 0:
                    # print("✅ DataLoader created with 0 workers (no test needed)")
                    break
                else:
                    # print("🧪 Testing DataLoader...")
                    pass
                    import signal

                    def timeout_handler(signum, frame):
                        raise TimeoutError("DataLoader test timed out")

                    # Set a 30-second timeout for the test
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)

                    if False:
                        try:
                            test_iter = iter(self.train_loader)
                            test_batch = next(test_iter)
                            del test_iter, test_batch  # Clean up
                            signal.alarm(0)  # Cancel timeout
                            signal.signal(signal.SIGALRM, old_handler)  # Restore handler
                            # print(f"✅ DataLoader test successful with {workers} workers")

                            if workers != num_workers:
                                # print(f"🔧 Reduced num_workers to {workers} to avoid issues")
                                pass
                            break
                        except TimeoutError:
                            signal.alarm(0)  # Cancel timeout
                            signal.signal(signal.SIGALRM, old_handler)  # Restore handler
                            # print(f"⏰ DataLoader test timed out with {workers} workers")
                            if workers == 0:
                                raise  # If even 0 workers fails, something is seriously wrong
                            continue  # Try with fewer workers

            except (OSError, RuntimeError, TimeoutError) as e:
                error_msg = str(e)
                if ("AF_UNIX path too long" in error_msg or
                    "timeout" in error_msg.lower() or
                    "deadlock" in error_msg.lower()) and workers > 0:
                    # print(f"⚠️  DataLoader issue with {workers} workers: {error_msg[:100]}...")
                    # print(f"   Trying {max(1, workers // 2) if workers > 1 else 0} workers")
                    continue
                else:
                    raise

        # Create validation and test loaders using proper splits
        val_batch_size = min(batch_size, 16)  # Smaller batches for validation

        # Validation loader
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=0,  # Use 0 workers for validation to avoid issues
            pin_memory=False,
            drop_last=False
        )

        # Test loader (optional, for future use)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=val_batch_size,
            shuffle=False,  # No shuffling for test
            num_workers=0,  # Use 0 workers for test to avoid issues
            pin_memory=False,
            drop_last=False
        )

        print(f"🔧 Data loaders created:")
        print(f"   📊 Train: {len(self.train_loader)} batches, {len(train_dataset):,} samples")
        print(f"   📊 Validation: {len(self.val_loader)} batches, {len(val_dataset):,} samples")
        print(f"   📊 Test: {len(self.test_loader)} batches, {len(test_dataset):,} samples")

        self.train_sampler = train_sampler
    
    def setup_training_components(self):
        """Setup optimizer, scheduler, EMA, and metrics."""
        # Optimizer
        self.optimizer = create_optimizer(
            self.model,
            learning_rate=self.config.world_size * self.config.training.get('learning_rate', 1e-4),
            weight_decay=self.config.training.get('weight_decay', 0.01)
        )

        # Scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            warmup_steps=self.config.training.get('warmup_steps', 1000),
            max_steps=self.config.training.get('max_steps', 100000)
        )

        # EMA
        self.ema_model = None
        if self.config.training.get('use_ema', True):
            self.ema_model = EMAModel(
                self.model.module if hasattr(self.model, 'module') else self.model,
                decay=self.config.training.get('ema_decay', 0.9999),
                device=self.device
            )

        # Metrics
        self.metrics = TrainingMetrics()

        # Training state
        self.current_step = 0
        self.start_time = time.time()
        self.accumulate_grad_batches = self.config.training.get('accumulate_grad_batches', 1)
        self.accumulation_step = 0

        # Sampling failure tracking
        self.sampling_failure_count = 0
        self.max_sampling_failures = self.config.training.get('max_sampling_failures', 3)
        self.stop_on_sampling_failure = self.config.training.get('stop_on_sampling_failure', True)

        # Load checkpoint if specified
        if self.config.resume_checkpoint is not None:
            self.load_training_checkpoint(self.config.resume_checkpoint)

        # Setup sampling output file
        self.setup_sampling_output()

        # print("🚀 Training components initialized")
        # print(f"📊 Gradient accumulation: {self.accumulate_grad_batches} batches")
        # print(f"🔍 Sampling failure handling: max_failures={self.max_sampling_failures}, stop_on_failure={self.stop_on_sampling_failure}")

    def setup_sampling_output(self):
        """Setup file for saving sampled sequences."""
        # Check if sampling file output is enabled
        sampling_config = getattr(self.config, 'sampling', None)
        save_to_file = True  # Default to True
        if sampling_config:
            save_to_file = getattr(sampling_config, 'save_to_file', True)

        self.save_sampling_to_file = save_to_file

        if not save_to_file:
            # print("📝 Sampling file output disabled")
            self.sampling_file = None
            return

        # Create sampling output directory
        sampling_dir = Path(self.config.work_dir) / "sampling"
        sampling_dir.mkdir(parents=True, exist_ok=True)

        # Create sampling output file with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sampling_file = sampling_dir / f"sampled_sequences_{timestamp}.txt"

        # Write header
        if is_main_process():
            with open(self.sampling_file, 'w') as f:
                f.write("# Sampled Protein Sequences During Training\n")
                f.write("# Format: STEP\tEPOCH\tSEQUENCE_ID\tSEQUENCE\n")
                f.write("# Generated on: {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                f.write("\n")

            # print(f"📝 Sampling output will be saved to: {self.sampling_file}")

    def save_sequences_to_file(self, sequences, step, epoch):
        """Save sampled sequences to file with step and epoch information."""
        if not is_main_process() or not sequences or not self.save_sampling_to_file or not self.sampling_file:
            pass

        try:
            with open(str(self.sampling_file), 'a') as f:
                for i, seq in enumerate(sequences):
                    # Clean up sequence (remove special tokens)
                    clean_seq = seq.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
                    if clean_seq:  # Only save non-empty sequences
                        f.write(f"{step}\t{epoch}\t{i+1}\t{clean_seq}\n")

            # print(f"📝 Saved {len(sequences)} sequences to {self.sampling_file}")
            pass

        except Exception as e:
            # print(f"⚠️  Failed to save sequences to file: {e}")
            pass

    def sample_sequences(self, step):
        """Sample sequences during training for monitoring."""
        # print(f"🔍 Debug: sample_sequences called at step {step}, is_main_process: {is_main_process()}")

        if not is_main_process():
            # print("not main process but still sampling")
            #return  # Only sample on main process
            pass

        try:
            # Get sampling config
            sampling_config = getattr(self.config, 'sampling', None)
            # print(f"🔍 Debug: sampling_config: {sampling_config}")

            if sampling_config is None:
                # print("🔍 Debug: No sampling config found, skipping sampling")
                return

            # Check if we should sample at this step
            sample_interval = getattr(sampling_config, 'sample_interval', 100)
            min_steps_before_sampling = getattr(sampling_config, 'min_steps_before_sampling', 500)
            # print(f"🔍 Debug: sample_interval: {sample_interval}, step % interval: {step % sample_interval}")

            # Skip sampling if we're too early in training (model not stable yet)
            if step < min_steps_before_sampling:
                # print(f"🔍 Debug: Skipping sampling - step {step} < min_steps_before_sampling {min_steps_before_sampling}")
                return

            if step % sample_interval != 0:
                return

            # print(f"\n🧬 Sampling sequences at step {step}...")

            # Use EMA model if available, otherwise use main model
            model_to_use = self.ema_model.ema_model if self.ema_model else self.model
            # print(f"🔍 Debug: Using model: {'EMA' if self.ema_model else 'main'}")

            # Sample sequences
            # print(f"🔍 Debug: Calling sample_during_training...")
            sequences = sample_during_training(
                model=model_to_use,
                graph=self.graph,
                noise=self.noise,
                config=self.config,
                step=step,
                device=self.device
            )
            # print(sequences)
            # print(f"🔍 Debug: Got {len(sequences) if sequences else 0} sequences")

            # Check for critical sampling failures
            if sequences is None:
                # print("❌ CRITICAL: Sampling returned None - this indicates a serious error!")
                self.sampling_failure_count += 1
                if self.stop_on_sampling_failure and self.sampling_failure_count >= self.max_sampling_failures:
                    # print(f"💥 STOPPING TRAINING: Too many consecutive sampling failures ({self.sampling_failure_count}/{self.max_sampling_failures})!")
                    raise RuntimeError(f"Training stopped due to {self.sampling_failure_count} consecutive sampling failures")
                return
            elif len(sequences) == 0:
                # print("⚠️  Warning: Sampling returned empty list")
                self.sampling_failure_count += 1
                if self.stop_on_sampling_failure and self.sampling_failure_count >= self.max_sampling_failures:
                    # print(f"💥 STOPPING TRAINING: Too many consecutive empty sampling results ({self.sampling_failure_count}/{self.max_sampling_failures})!")
                    raise RuntimeError(f"Training stopped due to {self.sampling_failure_count} consecutive empty sampling results")
                return
            else:
                # Reset failure counter on successful sampling
                if self.sampling_failure_count > 0:
                    # print(f"✅ Sampling recovered after {self.sampling_failure_count} failures")
                    pass
                self.sampling_failure_count = 0

            # Print sequences to console
            if sequences:
                # print(f"🧬 Generated {len(sequences)} sequences:")
                for i, seq in enumerate(sequences[:3]):  # Show first 3
                    clean_seq = seq.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
                    # print(f"  Sample {i+1}: {clean_seq[:80]}{'...' if len(clean_seq) > 80 else ''}")
                pass
            else:
                # print("🔍 Debug: No sequences generated")
                pass

            # Calculate current epoch (approximate)
            current_epoch = self.current_step // len(self.train_loader) if hasattr(self, 'train_loader') and len(self.train_loader) > 0 else 0

            # Save sequences to file
            self.save_sequences_to_file(sequences, step, current_epoch)

            # Log to wandb if available
            if self.wandb_run is not None and sequences:
                # print(f"🔍 Debug: Logging {len(sequences)} sequences to wandb")
                # Log first few sequences as text
                sample_text = "\n".join([f"Sample {i+1}: {seq.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()[:100]}..."
                                       for i, seq in enumerate(sequences[:3])])
                self.wandb_run.log({
                    "samples/sequences": sample_text,
                    "samples/step": step,
                    "samples/epoch": current_epoch
                }, step=step)
            else:
                # print(f"🔍 Debug: Not logging to wandb - wandb_run: {self.wandb_run is not None}, sequences: {len(sequences) if sequences else 0}")
                pass

        except Exception as e:
            # print(f"⚠️  Sampling failed at step {step}: {e}")
            import traceback
            # traceback.print_exc()

            # Track sampling failures and stop training if too many occur
            self.sampling_failure_count += 1
            # print(f"🔍 Sampling failure count: {self.sampling_failure_count}/{self.max_sampling_failures}")

            # Stop training if we have too many consecutive sampling failures
            if self.stop_on_sampling_failure and self.sampling_failure_count >= self.max_sampling_failures:
                # print(f"💥 STOPPING TRAINING: Too many consecutive sampling failures ({self.sampling_failure_count}/{self.max_sampling_failures})!")
                raise RuntimeError(f"Training stopped due to {self.sampling_failure_count} consecutive sampling failures: {e}")

            # For fewer failures, just log and continue
            # print(f"⚠️  Continuing training despite sampling failure ({self.sampling_failure_count}/{self.max_sampling_failures})")
            return
    
    def train_step(self, batch):
        """Execute a single training step with gradient accumulation support."""
        # print(f"🔄 train_step: Starting with batch type {type(batch)}, shape {batch.shape if hasattr(batch, 'shape') else 'no shape'}")

        # Move batch to device
        x0 = batch.to(self.device)
        batch_size = x0.shape[0]

        # Sample time and noise with rank-specific randomness
        # Use rank-specific generator to ensure different sigma values across ranks
        rank_generator = torch.Generator(device=self.device).manual_seed(
            self.config.seed + self.current_step * self.config.world_size + self.config.rank
        )
        t = torch.rand(batch_size, device=self.device, generator=rank_generator)
        sigma, dsigma = self.noise(t)

        # Debug: Print sigma shape and values
        # print(f"🔍 Sigma shape: {sigma.shape}, values: {sigma[:5]}")
        # print(f"🔍 DSigma shape: {dsigma.shape}, values: {dsigma[:5]}")
        # print(f"🔍 x0 shape: {x0.shape}")

        # Corrupt data
        xt = self.graph.sample_transition(x0, sigma)#[:, None])
        # print(f"🔍 xt shape after transition: {xt.shape}")

        # Debug: Compare x0 vs xt to verify corruption is working
        # print(f"🔍 Rank {self.config.rank}: x0 first sequence: {x0[0, :10]}")
        # print(f"🔍 Rank {self.config.rank}: xt first sequence: {xt[0, :10]}")
        # print(f"🔍 Rank {self.config.rank}: Corruption check - same?: {torch.equal(x0[0], xt[0])}")
        # print(f"🔍 Rank {self.config.rank}: Sigma for first sequence: {sigma[0]:.4f}")

        # Count how many tokens changed
        changed_tokens = (x0[0] != xt[0]).sum().item()
        total_tokens = x0[0].numel()
        corruption_rate = changed_tokens / total_tokens
        # print(f"🔍 Rank {self.config.rank}: Corruption rate: {changed_tokens}/{total_tokens} = {corruption_rate:.3f}")

        # Debug: Check if all ranks are seeing the same data (DDP issue)
        if self.config.world_size > 1:
            # Create a simple hash of the first sequence to compare across ranks
            x0_hash = hash(tuple(x0[0, :20].cpu().numpy()))
            # print(f"🔍 Rank {self.config.rank}: x0 hash (first 20 tokens): {x0_hash}")

            # Also check batch diversity within this rank
            if x0.shape[0] > 1:
                x0_hash_2 = hash(tuple(x0[1, :20].cpu().numpy()))
                same_within_batch = (x0_hash == x0_hash_2)
                # print(f"🔍 Rank {self.config.rank}: Same sequences within batch?: {same_within_batch}")
                if same_within_batch:
                    # print(f"🚨 Rank {self.config.rank}: WARNING - All sequences in batch are identical!")
                    pass
        # print(f"{xt.shape=}")
        # print(f"{x0.shape=}")

        # Forward pass
        if self.config.training.get('use_subs_loss', True):
            # SUBS loss
            model_output = self.model(xt, sigma, use_subs=True)

            # Debug: Check for NaN/Inf in model output during training
            if torch.any(torch.isnan(model_output)):
                # print(f"🚨 NaN detected in training model output at step {self.current_step}")
                # print(f"   Input xt shape: {xt.shape}, sigma shape: {sigma.shape}")
                # print(f"   xt range: [{torch.min(xt):.4f}, {torch.max(xt):.4f}]")
                # print(f"   sigma range: [{torch.min(sigma):.4f}, {torch.max(sigma):.4f}]")
                # print(f"   NaN count in output: {torch.sum(torch.isnan(model_output))}")

                # Check model parameters for NaN
                nan_params = []
                for name, param in self.model.named_parameters():
                    if torch.any(torch.isnan(param)):
                        nan_params.append(name)
                if nan_params:
                    # print(f"🚨 NaN detected in model parameters: {nan_params}")
                    pass

                # This is a critical error - we should probably stop training
                raise RuntimeError(f"Model produced NaN outputs at step {self.current_step}")

            if torch.any(torch.isinf(model_output)):
                # print(f"🚨 Inf detected in training model output at step {self.current_step}")
                # print(f"   Inf count in output: {torch.sum(torch.isinf(model_output))}")
                pass

            # Compute SUBS loss with curriculum learning
            loss, curric_dict = subs_loss_with_curriculum(
                model_output=model_output,
                x0=x0,
                sigma=sigma,
                noise_schedule=self.noise,
                training_step=self.current_step,
                #curriculum_config=self.config.curriculum
            )

            # Check loss for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                # print(f"🚨 Invalid loss detected at step {self.current_step}: {loss}")
                raise RuntimeError(f"Invalid loss at step {self.current_step}")

        else:
            # Original score-based loss (placeholder - implement if needed)
            model_output = self.model(xt, sigma, use_subs=False)
            loss = F.cross_entropy(model_output.view(-1, model_output.size(-1)), x0.view(-1))

        # Scale loss by accumulation steps for proper averaging
        scaled_loss = loss / self.accumulate_grad_batches

        # Backward pass
        scaled_loss.backward()

        # Compute metrics (use unscaled loss for logging)
        with torch.no_grad():
            accuracy = compute_accuracy(model_output, x0)
            perplexity = compute_perplexity(loss)
            # Get average sigma for this batch for monitoring
            avg_sigma = sigma.mean().item()

            # Debug high accuracy - print details every 100 steps
            if self.current_step % 100 == 0 and accuracy > 0.9:
                # print(f"\n🔍 High accuracy debug at step {self.current_step}:")
                # print(f"   Accuracy: {accuracy:.4f}")
                # print(f"   Avg sigma: {avg_sigma:.4f}")
                # print(f"   Sigma range: [{sigma.min().item():.4f}, {sigma.max().item():.4f}]")
                # print(f"   Model output shape: {model_output.shape}")
                # print(f"   Target shape: {x0.shape}")

                # Check if model is just predicting the same token everywhere
                pred_tokens = torch.argmax(model_output, dim=-1)
                unique_preds = torch.unique(pred_tokens).numel()
                unique_targets = torch.unique(x0).numel()
                # print(f"   Unique predicted tokens: {unique_preds}")
                # print(f"   Unique target tokens: {unique_targets}")

                # Check if predictions are too confident
                max_probs = torch.softmax(model_output, dim=-1).max(dim=-1)[0]
                avg_confidence = max_probs.mean().item()
                # print(f"   Average prediction confidence: {avg_confidence:.4f}")
                pass

        return loss.item(), accuracy, perplexity, avg_sigma
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = TrainingMetrics()
        loss = 100.0
        accuracy = 0.0
        perplexity = 100.0
        current_lr = 1.0e-5
        # Setup progress bar with timeout protection
        if True:
            pbar = tqdm(self.train_loader, desc=f"Training")#, postfix= {
                    #'loss': f"{loss:.4f}",
                    #'acc': f"{accuracy:.3f}",
                    #'ppl': f"{perplexity:.2f}",
                    #'lr': f"{current_lr:.2e}",
                    #'step': self.current_step
                #})
        #else:
        #    pbar = self.train_loader

        # Add distributed barrier to ensure all processes are ready
        if self.config.world_size > 1:
            try:
                dist.barrier(timeout=60)  # 1 minute timeout
            except Exception as e:
                # print(f"⚠️  Barrier timeout on rank {self.config.rank}: {e}")
                pass

        batch_count = 0
        last_log_time = time.time()

        try:
            for batch in pbar:
                batch_count += 1
                current_time = time.time()

                # Log progress every 30 seconds to detect hangs
                if current_time - last_log_time > 30:
                    # print(f"🔄 Rank {self.config.rank}: Processing batch {batch_count}, step {self.current_step}")
                    last_log_time = current_time
                step_start_time = time.time()

                # Training step with timeout protection
                try:
                    loss, accuracy, perplexity, avg_sigma = self.train_step(batch)
                except Exception as e:
                    # print(f"❌ Training step failed on rank {self.config.rank}: {e}")
                    # Skip this batch and continue
                    continue

                # Increment accumulation step
                self.accumulation_step += 1

                # Only perform optimization step when we've accumulated enough gradients
                if self.accumulation_step >= self.accumulate_grad_batches:
                    # Check for NaN gradients before clipping
                    nan_grad_params = []
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and torch.any(torch.isnan(param.grad)):
                            nan_grad_params.append(name)

                    if nan_grad_params:
                        # print(f"🚨 NaN gradients detected at step {self.current_step}: {nan_grad_params}")
                        # Zero out NaN gradients to prevent propagation
                        for name, param in self.model.named_parameters():
                            if param.grad is not None and torch.any(torch.isnan(param.grad)):
                                # print(f"   Zeroing NaN gradients in {name}")
                                param.grad.data[torch.isnan(param.grad.data)] = 0.0

                    # Gradient clipping and optimization
                    grad_norm = clip_gradients(
                        self.model,
                        max_norm=self.config.training.get('gradient_clip_norm', 1.0)
                    )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Update EMA
                    if self.ema_model is not None:
                        self.ema_model.update(self.model.module if hasattr(self.model, 'module') else self.model)

                    # Reset accumulation counter
                    self.accumulation_step = 0

                else:
                    # No optimization step, set grad_norm to 0 for logging
                    grad_norm = 0.0
                # Increment step counter (only after actual optimization step)
                self.current_step += 1


                # Update metrics
                step_time = time.time() - step_start_time
                current_lr = self.scheduler.get_last_lr()[0]

                self.metrics.update(
                    loss=loss,
                    accuracy=accuracy,
                    perplexity=perplexity,
                    avg_sigma=avg_sigma,
                    lr=current_lr,
                    grad_norm=grad_norm,
                    step_time=step_time
                )

                epoch_metrics.update(
                    loss=loss,
                    accuracy=accuracy,
                    perplexity=perplexity,
                    avg_sigma=avg_sigma,
                    lr=current_lr,
                    grad_norm=grad_norm,
                    step_time=step_time
                )

                # Update progress bar
                if True:  # is_main_process():
                    # Create a more informative progress bar
                    postfix_dict = {
                        'loss': f"{loss:.4f}",
                        'acc': f"{accuracy:.3f}",
                        'ppl': f"{perplexity:.2f}",
                        'σ': f"{avg_sigma:.3f}",  # Add sigma to progress bar
                        'lr': f"{current_lr:.2e}",
                        'step': self.current_step
                    }

                    # Only show accumulation step if we're accumulating gradients
                    if self.accumulate_grad_batches > 1:
                        postfix_dict['acc_step'] = f"{self.accumulation_step}/{self.accumulate_grad_batches}"

                    pbar.set_postfix(postfix_dict)

                # Log metrics periodically (only after actual optimization steps)
                if self.current_step % 20 == 0:  # and is_main_process():
                    self.log_training_metrics()

                # Sample sequences periodically
                sampling_config = getattr(self.config, 'sampling', None)
                sample_interval = getattr(sampling_config, 'sample_interval', 100) if sampling_config else 100
                if self.current_step % sample_interval == 0:  # and is_main_process():
                    try:
                        self.sample_sequences(self.current_step)
                    except RuntimeError as e:
                        if "sampling failures" in str(e):
                            print(f"💥 Training stopped due to sampling failures: {e}")
                            raise  # Re-raise to stop training
                        else:
                            # print(f"⚠️  Unexpected error in sampling: {e}")
                            # Continue training for other types of errors
                            pass

                # SUBS Validation every 500 steps (or configured frequency)
                current_val_loss = None
                if self.current_step % self.val_eval_freq == 0 and self.current_step > 0:
                    try:
                        print(f"\n🔍 Running SUBS validation at step {self.current_step}...")
                        current_val_loss, val_metrics = self.validate_model_subs()

                        # Update validation tracking
                        improved = self.update_validation_tracking(current_val_loss)

                        # Log validation metrics to wandb
                        if hasattr(self, 'wandb_run') and self.wandb_run is not None:
                            val_log_dict = {
                                'validation/loss': current_val_loss,
                                'validation/best_loss': self.best_val_loss,
                                'validation/steps_without_improvement': self.steps_without_improvement,
                                **val_metrics
                            }
                            self.wandb_run.log(val_log_dict, step=self.current_step)

                        # Check for early stopping
                        if self.steps_without_improvement >= self.patience:
                            print(f"🛑 Early stopping triggered after {self.patience} evaluations without improvement")
                            return epoch_metrics.get_averages()  # Exit training loop

                    except Exception as e:
                        print(f"⚠️  Validation failed at step {self.current_step}: {e}")
                        current_val_loss = None

                # Checkpointing every 1000 steps (or when validation improves)
                if (self.accumulation_step == 0 and
                    (self.current_step % self.checkpoint_freq == 0 or
                     (current_val_loss is not None and self.should_save_checkpoint(self.current_step, current_val_loss)))):

                    is_best = current_val_loss is not None and current_val_loss <= self.best_val_loss

                    # Only main process saves checkpoint
                    if is_main_process():
                        try:
                            self.save_training_checkpoint(
                                val_loss=current_val_loss,
                                is_best=is_best
                            )
                        except Exception as e:
                            print(f"⚠️  Failed to save checkpoint: {e}")

                    # Synchronize all processes after checkpointing
                    if self.config.world_size > 1:
                        try:
                            dist.barrier(timeout=120)  # 2 minute timeout for checkpoint sync
                        except Exception as e:
                            print(f"⚠️  Checkpoint barrier timeout on rank {self.config.rank}: {e}")

        except Exception as e:
            # print(f"❌ Training epoch failed on rank {self.config.rank}: {e}")
            # print(f"   Processed {batch_count} batches before failure")
            # Add barrier to ensure all processes are aware of the failure
            if self.config.world_size > 1:
                try:
                    dist.barrier(timeout=30)
                except:
                    pass
            raise

        finally:
            # Ensure progress bar is closed
            if hasattr(pbar, 'close'): #is_main_process() and 
                pbar.close()

        return epoch_metrics.get_averages()
    
    def log_training_metrics(self):
        """Log training metrics."""
        metrics = self.metrics.get_averages(window=100)

        # Add timing information
        elapsed_time = time.time() - self.start_time
        metrics['elapsed_time'] = elapsed_time
        metrics['steps_per_second'] = self.current_step / elapsed_time

        # Debug: Check if we have wandb
        # print(f"🔍 Debug: wandb_run exists: {hasattr(self, 'wandb_run')}, is not None: {getattr(self, 'wandb_run', None) is not None}")

        # Log to wandb if available
        if hasattr(self, 'wandb_run') and self.wandb_run is not None:
            # print(f"🔍 Debug: Logging metrics to wandb at step {self.current_step}: {list(metrics.keys())}")
            log_metrics(metrics, self.current_step, wandb_run=self.wandb_run)
        else:
            # print(f"🔍 Debug: No wandb logging - wandb_run: {getattr(self, 'wandb_run', 'not set')}")
            pass

        # Print summary only occasionally to reduce noise
        if self.current_step % 100 == 0:
            # print(f"\n📊 Step {self.current_step} | "
            #       f"Loss: {metrics.get('loss', 'N/A'):.4f} | "
            #       f"Acc: {metrics.get('accuracy', 0):.3f} | "
            #       f"PPL: {metrics.get('perplexity', 0):.2f} | "
            #       f"σ: {metrics.get('avg_sigma', 0):.3f} | "
            #       f"LR: {metrics.get('learning_rate', 0):.2e} | "
            #       f"Time: {format_time(elapsed_time)}")
            pass
    
    def validate_model_subs(self):
        """Run validation using SUBS loss on validation set."""
        print(f"🔍 Running SUBS validation...")

        self.model.eval()
        val_losses = []
        val_metrics = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= self.val_batch_limit:  # Limit validation batches for speed
                    break

                batch = batch.to(self.device)

                # Ensure batch is 2D: [batch_size, sequence_length]
                if batch.dim() != 2:
                    if batch.dim() > 2:
                        batch = batch.view(batch.shape[0], -1)
                    else:
                        print(f"⚠️  Skipping invalid batch with shape {batch.shape}")
                        continue

                try:
                    # Sample random timestep for each sequence in batch
                    # Use deterministic validation sampling (no rank dependency for reproducible validation)
                    val_generator = torch.Generator(device=self.device).manual_seed(
                        self.config.seed + i * 1000  # Use batch index for reproducible validation
                    )
                    t = torch.rand(batch.shape[0], device=self.device, generator=val_generator)
                    sigma, dsigma = self.noise(t)

                    # Add noise to create perturbed batch
                    perturbed_batch = self.graph.sample_transition(batch, sigma)

                    # Forward pass with SUBS parameterization
                    model_output = self.model(perturbed_batch, sigma, use_subs=True)

                    # Compute SUBS loss
                    from protlig_ddiff.processing.subs_loss import subs_loss, compute_subs_metrics
                    loss = subs_loss(model_output, batch, sigma, self.noise)
                    val_losses.append(loss.item())

                    # Compute additional metrics
                    metrics = compute_subs_metrics(model_output, batch, sigma)
                    val_metrics.append(metrics)

                except Exception as e:
                    print(f"⚠️  Error in validation batch {i}: {e}")
                    continue

        if not val_losses:
            print("❌ No valid validation batches processed!")
            return float('inf'), {}

        # Aggregate results
        avg_val_loss = np.mean(val_losses)

        # Aggregate metrics
        aggregated_metrics = {}
        if val_metrics:
            for key in val_metrics[0].keys():
                values = [m[key] for m in val_metrics if key in m]
                if values:
                    aggregated_metrics[f'val_{key}'] = np.mean(values)

        print(f"✅ Validation completed: {len(val_losses)} batches, avg loss: {avg_val_loss:.4f}")

        return avg_val_loss, aggregated_metrics

    def evaluate_test_set(self):
        """Run evaluation on test set (typically done at end of training)."""
        print(f"🔍 Running test set evaluation...")

        self.model.eval()
        test_losses = []
        test_metrics = []

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                batch = batch.to(self.device)

                # Ensure batch is 2D: [batch_size, sequence_length]
                if batch.dim() != 2:
                    if batch.dim() > 2:
                        batch = batch.view(batch.shape[0], -1)
                    else:
                        print(f"⚠️  Skipping invalid test batch with shape {batch.shape}")
                        continue

                try:
                    # Use deterministic test sampling (no rank dependency for reproducible test results)
                    test_generator = torch.Generator(device=self.device).manual_seed(
                        self.config.seed + i * 2000  # Use batch index for reproducible test evaluation
                    )
                    t = torch.rand(batch.shape[0], device=self.device, generator=test_generator)
                    sigma, dsigma = self.noise(t)
                    perturbed_batch = self.graph.sample_transition(batch, sigma)
                    model_output = self.model(perturbed_batch, sigma, use_subs=True)

                    from protlig_ddiff.processing.subs_loss import subs_loss, compute_subs_metrics
                    loss = subs_loss(model_output, batch, sigma, self.noise)
                    test_losses.append(loss.item())

                    metrics = compute_subs_metrics(model_output, batch, sigma)
                    test_metrics.append(metrics)

                except Exception as e:
                    print(f"⚠️  Error in test batch {i}: {e}")
                    continue

        if not test_losses:
            print("❌ No valid test batches processed!")
            return float('inf'), {}

        avg_test_loss = np.mean(test_losses)
        aggregated_metrics = {}
        if test_metrics:
            for key in test_metrics[0].keys():
                values = [m[key] for m in test_metrics if key in m]
                if values:
                    aggregated_metrics[f'test_{key}'] = np.mean(values)

        print(f"✅ Test evaluation completed: {len(test_losses)} batches, avg loss: {avg_test_loss:.4f}")

        # Log test results to wandb
        if hasattr(self, 'wandb_run') and self.wandb_run is not None:
            test_log_dict = {
                'test/final_loss': avg_test_loss,
                **aggregated_metrics
            }
            self.wandb_run.log(test_log_dict, step=self.current_step)

        return avg_test_loss, aggregated_metrics

    def should_save_checkpoint(self, step, val_loss=None):
        """Determine if we should save a checkpoint based on validation improvement."""
        # Always save at checkpoint frequency if not using improvement-based saving
        if not self.checkpoint_on_improvement:
            return step % self.checkpoint_freq == 0

        # If no validation loss provided, don't save
        if val_loss is None:
            return False

        # Check if this is an improvement
        improvement = (self.best_val_loss - val_loss) > self.min_delta

        # Save if it's an improvement or if we haven't saved in a while
        time_to_save = (step - self.last_checkpoint_step) >= self.checkpoint_freq

        return improvement or time_to_save

    def update_validation_tracking(self, val_loss):
        """Update validation loss tracking and early stopping."""
        self.val_loss_history.append(val_loss)

        # Check for improvement
        if (self.best_val_loss - val_loss) > self.min_delta:
            print(f"🎉 Validation improved: {self.best_val_loss:.4f} → {val_loss:.4f}")
            self.best_val_loss = val_loss
            self.steps_without_improvement = 0
            return True  # Improvement detected
        else:
            self.steps_without_improvement += 1
            print(f"📊 No improvement for {self.steps_without_improvement} evaluations (best: {self.best_val_loss:.4f})")
            return False  # No improvement

    def save_training_checkpoint(self, val_loss=None, is_best=False, force_save=False):
        """Save training checkpoint with validation information."""
        # Check if we should save this checkpoint
        if not force_save and not self.should_save_checkpoint(self.current_step, val_loss):
            return

        checkpoint_dir = Path(self.config.work_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_step_{self.current_step}.pt"

        # Enhanced checkpoint with validation info
        checkpoint = {
            'step': self.current_step,
            'best_loss': self.best_val_loss if val_loss is not None else float('inf'),
            'val_loss': val_loss,
            'val_loss_history': self.val_loss_history,
            'steps_without_improvement': self.steps_without_improvement,
            'model_state_dict': (self.model.module if hasattr(self.model, 'module') else self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': self.metrics.losses[-1] if self.metrics.losses else float('inf'),
        }

        if self.ema_model is not None:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()

        # Save checkpoint with timeout protection
        try:
            print(f"💾 Saving checkpoint: {checkpoint_path}")
            torch.save(checkpoint, checkpoint_path)
            self.last_checkpoint_step = self.current_step
            print(f"✅ Checkpoint saved successfully: {checkpoint_path}")

            if is_best:
                best_path = checkpoint_dir / 'best_checkpoint.pt'
                print(f"🏆 Saving best checkpoint: {best_path}")
                torch.save(checkpoint, best_path)
                print(f"✅ Best checkpoint saved: {best_path}")

            if val_loss is not None:
                print(f"   📊 Validation loss: {val_loss:.4f} (best: {self.best_val_loss:.4f})")

        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            # Don't raise the exception to avoid hanging the training
            import traceback
            traceback.print_exc()

    def load_training_checkpoint(self, checkpoint_path):
        """Load training checkpoint and restore validation state."""
        print(f"📂 Loading checkpoint from: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model state
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer and scheduler state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Load EMA if available
            if 'ema_state_dict' in checkpoint and self.ema_model is not None:
                self.ema_model.load_state_dict(checkpoint['ema_state_dict'])

            # Load training state
            self.current_step = checkpoint.get('step', 0)

            # Load validation tracking state
            self.best_val_loss = checkpoint.get('val_loss', checkpoint.get('best_loss', float('inf')))
            self.val_loss_history = checkpoint.get('val_loss_history', [])
            self.steps_without_improvement = checkpoint.get('steps_without_improvement', 0)

            print(f"✅ Checkpoint loaded successfully!")
            print(f"   📊 Restored step: {self.current_step}")
            print(f"   📊 Best validation loss: {self.best_val_loss:.4f}")
            print(f"   📊 Validation history length: {len(self.val_loss_history)}")
            print(f"   📊 Steps without improvement: {self.steps_without_improvement}")

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            raise
    
    def train(self, wandb_project=None, wandb_name=None):
        """Main training loop."""
        # Setup wandb with timeout protection
        if self.config.use_wandb :##and is_main_process():
            try:
                # print("🔧 Setting up Wandb...")
                import signal

                def wandb_timeout_handler(signum, frame):
                    raise TimeoutError("Wandb setup timed out")

                signal.signal(signal.SIGALRM, wandb_timeout_handler)
                signal.alarm(60)  # 1 minute timeout for wandb setup

                # Convert config to dict for wandb
                config_dict = {
                    'model': self.config.model.__dict__ if hasattr(self.config.model, '__dict__') else {},
                    'training': self.config.training.__dict__ if hasattr(self.config.training, '__dict__') else {},
                    'data': self.config.data.__dict__ if hasattr(self.config.data, '__dict__') else {},
                    'noise': self.config.noise.__dict__ if hasattr(self.config.noise, '__dict__') else {},
                    'tokens': self.config.tokens,
                    'devicetype': self.config.devicetype
                }
                self.wandb_run = setup_wandb(wandb_project, wandb_name, config_dict)
                signal.alarm(0)  # Cancel timeout
                # print("✅ Wandb setup successful")

            except (TimeoutError, Exception) as e:
                signal.alarm(0)
                # print(f"⚠️  Wandb setup failed: {e}")
                # print("   Continuing without wandb logging")
                self.wandb_run = None
        else:
            self.wandb_run = None
        
        # print(f"\n🚀 Starting training...")
        # print(f"📊 Total steps: {self.config.training.get('max_steps', 100000)}")

        try:
            # Training loop
            max_steps = self.config.training.get('max_steps', 100000)
            epoch_count = 0
            consecutive_failures = 0
            max_consecutive_failures = 3

            while self.current_step < max_steps:
                epoch_count += 1
                epoch_start_time = time.time()

                try:
                    # Set epoch for distributed sampler
                    if self.train_sampler is not None:
                        epoch = self.current_step // len(self.train_loader)
                        # print(f"🔍 Rank {self.config.rank}: Setting sampler epoch to {epoch}")
                        self.train_sampler.set_epoch(epoch)
                        # print(f"🔍 Rank {self.config.rank}: Sampler epoch set, will see {len(self.train_sampler)} samples")

                    # print(f"\n🚀 Rank {self.config.rank}: Starting epoch {epoch_count}, step {self.current_step}")

                    # Train epoch with timeout detection
                    epoch_metrics = self.train_epoch()

                    epoch_time = time.time() - epoch_start_time
                    consecutive_failures = 0  # Reset failure counter on success

                    if is_main_process():
                        # print(f"\n✅ Epoch {epoch_count} completed in {epoch_time:.1f}s | Avg Loss: {epoch_metrics['loss']:.4f}")
                        pass

                    # Check if we've reached max steps
                    if self.current_step >= max_steps:
                        break

                except Exception as e:
                    consecutive_failures += 1
                    # print(f"❌ Epoch {epoch_count} failed on rank {self.config.rank}: {e}")

                    if consecutive_failures >= max_consecutive_failures:
                        # print(f"💥 Too many consecutive failures ({consecutive_failures}), stopping training")
                        break
                    else:
                        # print(f"🔄 Retrying... ({consecutive_failures}/{max_consecutive_failures} failures)")
                        time.sleep(5)  # Brief pause before retry
                        continue
            
            # Final checkpoint and test evaluation
            if is_main_process():
                self.save_training_checkpoint(force_save=True)

                # Run final test evaluation
                try:
                    print(f"\n🧪 Running final test set evaluation...")
                    test_loss, test_metrics = self.evaluate_test_set()
                    print(f"🏁 Final test loss: {test_loss:.4f}")
                    for metric, value in test_metrics.items():
                        print(f"   📊 {metric}: {value:.4f}")
                except Exception as e:
                    print(f"⚠️  Test evaluation failed: {e}")

                # print(f"\n🎉 Training completed! Final step: {self.current_step}")

            # Synchronize all processes after final checkpoint and evaluation
            if self.config.world_size > 1:
                try:
                    dist.barrier(timeout=180)  # 3 minute timeout for final sync
                    print(f"✅ Rank {self.config.rank}: Final synchronization completed")
                except Exception as e:
                    print(f"⚠️  Final barrier timeout on rank {self.config.rank}: {e}")

        finally:
            # Cleanup with error handling
            # print(f"\n🧹 Cleaning up on rank {self.config.rank}...")

            # Cleanup wandb
            if self.wandb_run is not None:
                try:
                    self.wandb_run.finish()
                    # print("✅ Wandb cleanup completed")
                except Exception as e:
                    # print(f"⚠️  Wandb cleanup failed: {e}")
                    pass

            # Cleanup DDP
            if self.config.world_size > 1:
                try:
                    # Add barrier before cleanup to ensure all processes are ready
                    dist.barrier(timeout=30)
                    cleanup_ddp()
                    # print("✅ DDP cleanup completed")
                except Exception as e:
                    # print(f"⚠️  DDP cleanup failed: {e}")
                    pass

            # Cleanup temporary directory if we created it
            try:
                cleanup_temp_directory()
            except Exception as e:
                # print(f"⚠️  Temp directory cleanup failed: {e}")
                pass

            # print(f"🏁 Cleanup completed on rank {self.config.rank}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clean UniRef50 Discrete Diffusion Training")
    
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--datafile", type=str, required=True, help="Path to training data")
    parser.add_argument("--work_dir", type=str, default="./work_dir", help="Working directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (e.g., 'cpu', '0', 'cuda:0', 'xpu:0')")
    parser.add_argument("--devicetype", type=str, default="xpu", help="Device to use (e.g., 'cpu', '0', 'cuda:0', 'xpu:0')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--resume_checkpoint", type=str, help="Resume from checkpoint")
    parser.add_argument("--wandb_project", type=str, default="sedd-training", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, help="Wandb run name")

    # DDP arguments
    parser.add_argument("--cluster", type=str, choices=["aurora", "polaris"], help="Cluster type for DDP (omit for single GPU)")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # print("\n" + "="*80)
    # print("🧬 CLEAN UNIREF50 DISCRETE DIFFUSION TRAINING")
    # print("="*80)
    
    try:
        # Setup DDP if specified
        rank, world_size, device = 0, 1, args.device
        
        print(f"Cluster type: {args.cluster}")
        if args.cluster == "aurora":
            print("Setting up Aurora DDP...")
            try:
                rank, device, world_size = setup_ddp_aurora()
                print(f"Aurora DDP setup successful: rank={rank}, device={device}, world_size={world_size}")
            except Exception as e:
                print(f"Aurora DDP setup failed: {e}")
                import traceback
                traceback.print_exc()
                raise
        elif args.cluster == "polaris":
            print("Setting up Polaris DDP...")
            try:
                rank, device, world_size = setup_ddp_polaris(rank, world_size)
                print(f"Polaris DDP setup successful: rank={rank}, device={device}, world_size={world_size}")
            except Exception as e:
                print(f"Polaris DDP setup failed: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # Create trainer config
        print("Creating trainer config...")
        trainer_config = TrainerConfig(
            work_dir=args.work_dir,
            config_file=args.config,
            datafile=args.datafile,
            rank=rank,
            world_size=world_size,
            device=device,
            devicetype=args.devicetype,
            seed=args.seed,
            use_wandb=not args.no_wandb,
            resume_checkpoint=args.resume_checkpoint
        )
        print(f"Trainer config created: rank={rank}, world_size={world_size}, device={device}")

        # Create and run trainer
        print("Creating trainer...")
        trainer = UniRef50Trainer(trainer_config)
        print("Trainer created, starting training...")
        trainer.train(args.wandb_project, args.wandb_name)
        
        # print("\n🎉 Training completed successfully!")
        return 0

    except Exception as e:
        # print(f"\n❌ Training failed: {e}")
        import traceback
        # traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
