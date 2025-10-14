"""
Clean and organized training script for UniRef50 discrete diffusion training.
"""
from mpi4py import MPI
print("loaded MPI")
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
                print(f"🔧 Set temporary directory to: {temp_dir}")
                return temp_dir
            except Exception as e:
                print(f"⚠️  Failed to create temp dir in {tmp_dir}: {e}")
                continue

    # If all else fails, use current directory
    current_tmp = os.path.join(os.getcwd(), 'tmp_pytorch')
    os.makedirs(current_tmp, exist_ok=True)
    os.environ['TMPDIR'] = current_tmp
    os.environ['TEMP'] = current_tmp
    os.environ['TMP'] = current_tmp
    print(f"🔧 Set temporary directory to: {current_tmp}")
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
            print(f"🧹 Cleaned up temporary directory: {_temp_dir_created}")
        except Exception as e:
            print(f"⚠️  Failed to cleanup temp directory {_temp_dir_created}: {e}")

import torch
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
    print(SIZE, RANK)
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

    print(f"DDP: Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK}. {MASTER_ADDR}")

    # Initialize distributed communication
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Set CUDA device
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    
    return dist.get_rank(), device_id, dist.get_world_size()

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
    device: str = 'cpu'
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
    devicetype: str = 'cuda'

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
                print("⚠️  CUDA not available, falling back to CPU")
                self.device = torch.device('cpu')
            elif self.device.index is not None and self.device.index >= torch.cuda.device_count():
                print(f"⚠️  GPU {self.device.index} not available, using GPU 0")
                self.device = torch.device('cuda:0')

        print(f"🔧 Environment setup: device={self.device}, seed={self.config.seed}")

        # Set CUDA device if using GPU
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        
    def setup_model_and_data(self):
        """Setup model, data, and related components."""
        # Print config summary
        if is_main_process():
            print("📋 Configuration Summary:")
            print(f"  Model: dim={self.config.model.dim}, layers={self.config.model.n_layers}, heads={self.config.model.n_heads}")
            print(f"  Training: lr={self.config.training.learning_rate}, batch_size={self.config.training.batch_size}")
            print(f"  Data: max_length={self.config.data.max_length}, tokenize_on_fly={self.config.data.tokenize_on_fly}")
            print(f"  Noise: type={self.config.noise.type}")

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
            print(f"⚠️  Unknown noise type: {noise_type}, defaulting to LogLinear")
            self.noise = noise_lib.LogLinearNoise()

        # Setup model - pass the config directly
        self.model = DiscDiffModel(self.config).to(self.device)
        
        # Setup DDP if needed
        if self.config.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.device] if self.device.type == 'cuda' else None)
        
        # Setup data
        self.setup_data_loaders()
        
        print(f"🏗️  Model setup: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"📊 Data setup: {len(self.train_loader)} batches per epoch")
    
    def setup_data_loaders(self):
        """Setup training and validation data loaders."""
        # Set multiprocessing start method to avoid issues on HPC systems
        import multiprocessing as mp
        try:
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)
                print("🔧 Set multiprocessing start method to 'spawn'")
        except RuntimeError as e:
            print(f"⚠️  Could not set multiprocessing start method: {e}")

        # Add timeout for data loading operations
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Data loading operation timed out")

        # Training dataset with timeout protection
        print("📂 Loading training dataset...")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 minute timeout for dataset loading

        try:
            # Get data settings from config
            tokenize_on_fly = self.config.data.get('tokenize_on_fly', True)  # Default to True for safety
            use_streaming = self.config.data.get('use_streaming', True)      # Default to True for large datasets
            max_length = self.config.data.get('max_length', 256)            # Shorter default for tokenization

            # Disable streaming for .pt files (binary format doesn't support streaming)
            if self.config.datafile.endswith('.pt'):
                use_streaming = False
                print("🔧 Disabled streaming for .pt file (binary format)")

            print(f"🔧 Dataset settings: tokenize_on_fly={tokenize_on_fly}, use_streaming={use_streaming}, max_length={max_length}")

            train_dataset = UniRef50Dataset(
                data_file=self.config.datafile,
                tokenize_on_fly=tokenize_on_fly,
                max_length=max_length,
                use_streaming=use_streaming
            )
            signal.alarm(0)  # Cancel timeout
            print(f"✅ Dataset loaded successfully: {len(train_dataset)} samples")
        except TimeoutError:
            signal.alarm(0)
            print("⚠️  Dataset loading timed out, trying with streaming=True")
            train_dataset = UniRef50Dataset(
                data_file=self.config.datafile,
                tokenize_on_fly=self.config.data.get('tokenize_on_fly', False),
                max_length=self.config.data.get('max_length', 512),
                use_streaming=True  # Force streaming on timeout
            )
        
        # Setup sampler for DDP
        train_sampler = None
        if self.config.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=True
            )
        
        # Training data loader with fallback for num_workers
        num_workers = self.config.data.get('num_workers', 4)

        # For HPC systems, start with fewer workers to avoid hangs
        if self.config.world_size > 1:
            num_workers = min(num_workers, 2)  # Limit workers in distributed mode
            print(f"🔧 Limited num_workers to {num_workers} for distributed training")

        # For large datasets, start with fewer workers to avoid memory issues
        dataset_size = len(train_dataset)
        if dataset_size > 1000000:  # 1M+ sequences
            num_workers = min(num_workers, 1)  # Start with 1 worker for large datasets
            print(f"🔧 Large dataset detected ({dataset_size:,} samples), starting with {num_workers} worker(s)")

        # Try to create data loader, reduce num_workers if issues occur
        for workers in [num_workers, max(1, num_workers // 2), 1, 0]:
            try:
                print(f"🔧 Trying DataLoader with {workers} workers...")
                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config.training.get('batch_size', 32),
                    shuffle=(train_sampler is None),
                    sampler=train_sampler,
                    num_workers=workers,
                    pin_memory=self.config.data.get('pin_memory', True) and workers > 0,
                    drop_last=True,
                    timeout=30 if workers > 0 else 0,  # Add timeout for worker processes
                    persistent_workers=False  # Disable persistent workers to avoid hangs
                )

                # Test the data loader by getting one batch
                if workers == 0:
                    print("✅ DataLoader created with 0 workers (no test needed)")
                    break
                else:
                    print("🧪 Testing DataLoader...")
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
                            print(f"✅ DataLoader test successful with {workers} workers")

                            if workers != num_workers:
                                print(f"🔧 Reduced num_workers to {workers} to avoid issues")
                            break
                        except TimeoutError:
                            signal.alarm(0)  # Cancel timeout
                            signal.signal(signal.SIGALRM, old_handler)  # Restore handler
                            print(f"⏰ DataLoader test timed out with {workers} workers")
                            if workers == 0:
                                raise  # If even 0 workers fails, something is seriously wrong
                            continue  # Try with fewer workers

            except (OSError, RuntimeError, TimeoutError) as e:
                error_msg = str(e)
                if ("AF_UNIX path too long" in error_msg or
                    "timeout" in error_msg.lower() or
                    "deadlock" in error_msg.lower()) and workers > 0:
                    print(f"⚠️  DataLoader issue with {workers} workers: {error_msg[:100]}...")
                    print(f"   Trying {max(1, workers // 2) if workers > 1 else 0} workers")
                    continue
                else:
                    raise
        
        self.train_sampler = train_sampler
    
    def setup_training_components(self):
        """Setup optimizer, scheduler, EMA, and metrics."""
        # Optimizer
        self.optimizer = create_optimizer(
            self.model,
            learning_rate=self.config.training.get('learning_rate', 1e-4),
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

        # Setup sampling output file
        self.setup_sampling_output()

        print("🚀 Training components initialized")
        print(f"📊 Gradient accumulation: {self.accumulate_grad_batches} batches")
        print(f"🔍 Sampling failure handling: max_failures={self.max_sampling_failures}, stop_on_failure={self.stop_on_sampling_failure}")

    def setup_sampling_output(self):
        """Setup file for saving sampled sequences."""
        # Check if sampling file output is enabled
        sampling_config = getattr(self.config, 'sampling', None)
        save_to_file = True  # Default to True
        if sampling_config:
            save_to_file = getattr(sampling_config, 'save_to_file', True)

        self.save_sampling_to_file = save_to_file

        if not save_to_file:
            print("📝 Sampling file output disabled")
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

            print(f"📝 Sampling output will be saved to: {self.sampling_file}")

    def save_sequences_to_file(self, sequences, step, epoch):
        """Save sampled sequences to file with step and epoch information."""
        if not is_main_process() or not sequences or not self.save_sampling_to_file or not self.sampling_file:
            pass

        try:
            with open(self.sampling_file, 'a') as f:
                for i, seq in enumerate(sequences):
                    # Clean up sequence (remove special tokens)
                    clean_seq = seq.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
                    if clean_seq:  # Only save non-empty sequences
                        f.write(f"{step}\t{epoch}\t{i+1}\t{clean_seq}\n")

            print(f"📝 Saved {len(sequences)} sequences to {self.sampling_file}")

        except Exception as e:
            print(f"⚠️  Failed to save sequences to file: {e}")

    def sample_sequences(self, step):
        """Sample sequences during training for monitoring."""
        print(f"🔍 Debug: sample_sequences called at step {step}, is_main_process: {is_main_process()}")

        if not is_main_process():
            print("not main process but still sampling")
            #return  # Only sample on main process

        try:
            # Get sampling config
            sampling_config = getattr(self.config, 'sampling', None)
            print(f"🔍 Debug: sampling_config: {sampling_config}")

            if sampling_config is None:
                print("🔍 Debug: No sampling config found, skipping sampling")
                return

            # Check if we should sample at this step
            sample_interval = getattr(sampling_config, 'sample_interval', 100)
            min_steps_before_sampling = getattr(sampling_config, 'min_steps_before_sampling', 500)
            print(f"🔍 Debug: sample_interval: {sample_interval}, step % interval: {step % sample_interval}")

            # Skip sampling if we're too early in training (model not stable yet)
            if step < min_steps_before_sampling:
                print(f"🔍 Debug: Skipping sampling - step {step} < min_steps_before_sampling {min_steps_before_sampling}")
                return

            if step % sample_interval != 0:
                return

            print(f"\n🧬 Sampling sequences at step {step}...")

            # Use EMA model if available, otherwise use main model
            model_to_use = self.ema_model.ema_model if self.ema_model else self.model
            print(f"🔍 Debug: Using model: {'EMA' if self.ema_model else 'main'}")

            # Sample sequences
            print(f"🔍 Debug: Calling sample_during_training...")
            sequences = sample_during_training(
                model=model_to_use,
                graph=self.graph,
                noise=self.noise,
                config=self.config,
                step=step,
                device=self.device
            )
            print(sequences)
            print(f"🔍 Debug: Got {len(sequences) if sequences else 0} sequences")

            # Check for critical sampling failures
            if sequences is None:
                print("❌ CRITICAL: Sampling returned None - this indicates a serious error!")
                self.sampling_failure_count += 1
                if self.stop_on_sampling_failure and self.sampling_failure_count >= self.max_sampling_failures:
                    print(f"💥 STOPPING TRAINING: Too many consecutive sampling failures ({self.sampling_failure_count}/{self.max_sampling_failures})!")
                    raise RuntimeError(f"Training stopped due to {self.sampling_failure_count} consecutive sampling failures")
                return
            elif len(sequences) == 0:
                print("⚠️  Warning: Sampling returned empty list")
                self.sampling_failure_count += 1
                if self.stop_on_sampling_failure and self.sampling_failure_count >= self.max_sampling_failures:
                    print(f"💥 STOPPING TRAINING: Too many consecutive empty sampling results ({self.sampling_failure_count}/{self.max_sampling_failures})!")
                    raise RuntimeError(f"Training stopped due to {self.sampling_failure_count} consecutive empty sampling results")
                return
            else:
                # Reset failure counter on successful sampling
                if self.sampling_failure_count > 0:
                    print(f"✅ Sampling recovered after {self.sampling_failure_count} failures")
                self.sampling_failure_count = 0

            # Print sequences to console
            if sequences:
                print(f"🧬 Generated {len(sequences)} sequences:")
                for i, seq in enumerate(sequences[:3]):  # Show first 3
                    clean_seq = seq.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
                    print(f"  Sample {i+1}: {clean_seq[:80]}{'...' if len(clean_seq) > 80 else ''}")
            else:
                print("🔍 Debug: No sequences generated")

            # Calculate current epoch (approximate)
            current_epoch = self.current_step // len(self.train_loader) if hasattr(self, 'train_loader') and len(self.train_loader) > 0 else 0

            # Save sequences to file
            self.save_sequences_to_file(sequences, step, current_epoch)

            # Log to wandb if available
            if self.wandb_run is not None and sequences:
                print(f"🔍 Debug: Logging {len(sequences)} sequences to wandb")
                # Log first few sequences as text
                sample_text = "\n".join([f"Sample {i+1}: {seq.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()[:100]}..."
                                       for i, seq in enumerate(sequences[:3])])
                self.wandb_run.log({
                    "samples/sequences": sample_text,
                    "samples/step": step,
                    "samples/epoch": current_epoch
                }, step=step)
            else:
                print(f"🔍 Debug: Not logging to wandb - wandb_run: {self.wandb_run is not None}, sequences: {len(sequences) if sequences else 0}")

        except Exception as e:
            print(f"⚠️  Sampling failed at step {step}: {e}")
            import traceback
            traceback.print_exc()

            # Track sampling failures and stop training if too many occur
            self.sampling_failure_count += 1
            print(f"🔍 Sampling failure count: {self.sampling_failure_count}/{self.max_sampling_failures}")

            # Stop training if we have too many consecutive sampling failures
            if self.stop_on_sampling_failure and self.sampling_failure_count >= self.max_sampling_failures:
                print(f"💥 STOPPING TRAINING: Too many consecutive sampling failures ({self.sampling_failure_count}/{self.max_sampling_failures})!")
                raise RuntimeError(f"Training stopped due to {self.sampling_failure_count} consecutive sampling failures: {e}")

            # For fewer failures, just log and continue
            print(f"⚠️  Continuing training despite sampling failure ({self.sampling_failure_count}/{self.max_sampling_failures})")
            return
    
    def train_step(self, batch):
        """Execute a single training step with gradient accumulation support."""
        #print(f"🔄 train_step: Starting with batch type {type(batch)}, shape {batch.shape if hasattr(batch, 'shape') else 'no shape'}")

        # Move batch to device
        x0 = batch.to(self.device)
        batch_size = x0.shape[0]

        # Sample time and noise
        t = torch.rand(batch_size, device=self.device)
        sigma, dsigma = self.noise(t)

        # Corrupt data
        xt = self.graph.sample_transition(x0, sigma)#[:, None])
        #print(f"{xt.shape=}")
        #print(f"{x0.shape=}")

        # Forward pass
        if self.config.training.get('use_subs_loss', True):
            # SUBS loss
            model_output = self.model(xt, sigma, use_subs=True)

            # Debug: Check for NaN/Inf in model output during training
            if torch.any(torch.isnan(model_output)):
                print(f"🚨 NaN detected in training model output at step {self.current_step}")
                print(f"   Input xt shape: {xt.shape}, sigma shape: {sigma.shape}")
                print(f"   xt range: [{torch.min(xt):.4f}, {torch.max(xt):.4f}]")
                print(f"   sigma range: [{torch.min(sigma):.4f}, {torch.max(sigma):.4f}]")
                print(f"   NaN count in output: {torch.sum(torch.isnan(model_output))}")

                # Check model parameters for NaN
                nan_params = []
                for name, param in self.model.named_parameters():
                    if torch.any(torch.isnan(param)):
                        nan_params.append(name)
                if nan_params:
                    print(f"🚨 NaN detected in model parameters: {nan_params}")

                # This is a critical error - we should probably stop training
                raise RuntimeError(f"Model produced NaN outputs at step {self.current_step}")

            if torch.any(torch.isinf(model_output)):
                print(f"🚨 Inf detected in training model output at step {self.current_step}")
                print(f"   Inf count in output: {torch.sum(torch.isinf(model_output))}")

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
                print(f"🚨 Invalid loss detected at step {self.current_step}: {loss}")
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

        return loss.item(), accuracy, perplexity
    
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
                print(f"⚠️  Barrier timeout on rank {self.config.rank}: {e}")

        batch_count = 0
        last_log_time = time.time()

        try:
            for batch in pbar:
                batch_count += 1
                current_time = time.time()

                # Log progress every 30 seconds to detect hangs
                if current_time - last_log_time > 30:
                    print(f"🔄 Rank {self.config.rank}: Processing batch {batch_count}, step {self.current_step}")
                    last_log_time = current_time
                step_start_time = time.time()

                # Training step with timeout protection
                try:
                    loss, accuracy, perplexity = self.train_step(batch)
                except Exception as e:
                    print(f"❌ Training step failed on rank {self.config.rank}: {e}")
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
                        print(f"🚨 NaN gradients detected at step {self.current_step}: {nan_grad_params}")
                        # Zero out NaN gradients to prevent propagation
                        for name, param in self.model.named_parameters():
                            if param.grad is not None and torch.any(torch.isnan(param.grad)):
                                print(f"   Zeroing NaN gradients in {name}")
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
                    lr=current_lr,
                    grad_norm=grad_norm,
                    step_time=step_time
                )

                epoch_metrics.update(
                    loss=loss,
                    accuracy=accuracy,
                    perplexity=perplexity,
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
                            print(f"⚠️  Unexpected error in sampling: {e}")
                            # Continue training for other types of errors

                    # Save checkpoint periodically (only after actual optimization steps)
                    if self.accumulation_step == 0 and self.current_step % 5000 == 0:  # and is_main_process():
                        try:
                            self.save_training_checkpoint()
                        except Exception as e:
                            print(f"⚠️  Failed to save checkpoint: {e}")

        except Exception as e:
            print(f"❌ Training epoch failed on rank {self.config.rank}: {e}")
            print(f"   Processed {batch_count} batches before failure")
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
        print(f"🔍 Debug: wandb_run exists: {hasattr(self, 'wandb_run')}, is not None: {getattr(self, 'wandb_run', None) is not None}")

        # Log to wandb if available
        if hasattr(self, 'wandb_run') and self.wandb_run is not None:
            print(f"🔍 Debug: Logging metrics to wandb at step {self.current_step}: {list(metrics.keys())}")
            log_metrics(metrics, self.current_step, wandb_run=self.wandb_run)
        else:
            print(f"🔍 Debug: No wandb logging - wandb_run: {getattr(self, 'wandb_run', 'not set')}")

        # Print summary only occasionally to reduce noise
        if self.current_step % 100 == 0:
            print(f"\n📊 Step {self.current_step} | "
                  f"Loss: {metrics.get('loss', 'N/A'):.4f} | "
                  f"Acc: {metrics.get('accuracy', 0):.3f} | "
                  f"PPL: {metrics.get('perplexity', 0):.2f} | "
                  f"LR: {metrics.get('learning_rate', 0):.2e} | "
                  f"Time: {format_time(elapsed_time)}")
    
    def save_training_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.work_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{self.current_step}.pt"
        
        save_checkpoint(
            model=self.model.module if hasattr(self.model, 'module') else self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema_model=self.ema_model,
            step=self.current_step,
            loss=self.metrics.losses[-1] if self.metrics.losses else float('inf'),
            save_path=checkpoint_path
        )
    
    def train(self, wandb_project=None, wandb_name=None):
        """Main training loop."""
        # Setup wandb with timeout protection
        if self.config.use_wandb :##and is_main_process():
            try:
                print("🔧 Setting up Wandb...")
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
                print("✅ Wandb setup successful")

            except (TimeoutError, Exception) as e:
                signal.alarm(0)
                print(f"⚠️  Wandb setup failed: {e}")
                print("   Continuing without wandb logging")
                self.wandb_run = None
        else:
            self.wandb_run = None
        
        print(f"\n🚀 Starting training...")
        print(f"📊 Total steps: {self.config.training.get('max_steps', 100000)}")

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
                        self.train_sampler.set_epoch(epoch)

                    print(f"\n🚀 Starting epoch {epoch_count}, step {self.current_step}")

                    # Train epoch with timeout detection
                    epoch_metrics = self.train_epoch()

                    epoch_time = time.time() - epoch_start_time
                    consecutive_failures = 0  # Reset failure counter on success

                    if is_main_process():
                        print(f"\n✅ Epoch {epoch_count} completed in {epoch_time:.1f}s | Avg Loss: {epoch_metrics['loss']:.4f}")

                    # Check if we've reached max steps
                    if self.current_step >= max_steps:
                        break

                except Exception as e:
                    consecutive_failures += 1
                    print(f"❌ Epoch {epoch_count} failed on rank {self.config.rank}: {e}")

                    if consecutive_failures >= max_consecutive_failures:
                        print(f"💥 Too many consecutive failures ({consecutive_failures}), stopping training")
                        break
                    else:
                        print(f"🔄 Retrying... ({consecutive_failures}/{max_consecutive_failures} failures)")
                        time.sleep(5)  # Brief pause before retry
                        continue
            
            # Final checkpoint
            if is_main_process():
                self.save_training_checkpoint()
                print(f"\n🎉 Training completed! Final step: {self.current_step}")
        
        finally:
            # Cleanup with error handling
            print(f"\n🧹 Cleaning up on rank {self.config.rank}...")

            # Cleanup wandb
            if self.wandb_run is not None:
                try:
                    self.wandb_run.finish()
                    print("✅ Wandb cleanup completed")
                except Exception as e:
                    print(f"⚠️  Wandb cleanup failed: {e}")

            # Cleanup DDP
            if self.config.world_size > 1:
                try:
                    # Add barrier before cleanup to ensure all processes are ready
                    dist.barrier(timeout=30)
                    cleanup_ddp()
                    print("✅ DDP cleanup completed")
                except Exception as e:
                    print(f"⚠️  DDP cleanup failed: {e}")

            # Cleanup temporary directory if we created it
            try:
                cleanup_temp_directory()
            except Exception as e:
                print(f"⚠️  Temp directory cleanup failed: {e}")

            print(f"🏁 Cleanup completed on rank {self.config.rank}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clean UniRef50 Discrete Diffusion Training")
    
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--datafile", type=str, required=True, help="Path to training data")
    parser.add_argument("--work_dir", type=str, default="./work_dir", help="Working directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (e.g., 'cpu', '0', 'cuda:0')")
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
    
    print("\n" + "="*80)
    print("🧬 CLEAN UNIREF50 DISCRETE DIFFUSION TRAINING")
    print("="*80)
    
    try:
        # Setup DDP if specified
        rank, world_size, device = 0, 1, args.device
        
        if args.cluster == "aurora":
            rank, device, world_size = setup_ddp_aurora()
        elif args.cluster == "polaris":
            rank, device, world_size = setup_ddp_polaris(rank, world_size)
        
        # Create trainer config
        trainer_config = TrainerConfig(
            work_dir=args.work_dir,
            config_file=args.config,
            datafile=args.datafile,
            rank=rank,
            world_size=world_size,
            device=device,
            seed=args.seed,
            use_wandb=not args.no_wandb,
            resume_checkpoint=args.resume_checkpoint
        )
        
        # Create and run trainer
        trainer = UniRef50Trainer(trainer_config)
        trainer.train(args.wandb_project, args.wandb_name)
        
        print("\n🎉 Training completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
