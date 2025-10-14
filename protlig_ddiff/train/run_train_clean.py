"""
Clean and organized training script for UniRef50 discrete diffusion training.
"""
import os
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import utilities
from protlig_ddiff.utils.ddp_utils import setup_ddp_aurora, setup_ddp_polaris, cleanup_ddp, is_main_process, get_rank, get_world_size
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


@dataclass
class TrainerConfig:
    """Configuration for the trainer."""
    work_dir: str
    config_file: str
    datafile: str
    rank: int = 0
    world_size: int = 1
    device: str = 'cpu'
    seed: int = 42
    use_wandb: bool = True
    resume_checkpoint: Optional[str] = None


class UniRef50Trainer:
    """Clean and organized trainer for UniRef50 discrete diffusion training."""
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.setup_environment()
        self.load_training_config()
        self.setup_model_and_data()
        self.setup_training_components()
        
    def setup_environment(self):
        """Setup training environment and device."""
        # Set random seeds
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        # Setup device
        if self.config.device.startswith('xpu'):
            self.device = torch.device(self.config.device)
        elif self.config.device.startswith('cuda'):
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device('cpu')
        
        print(f"🔧 Environment setup: device={self.device}, seed={self.config.seed}")
        
    def load_training_config(self):
        """Load and parse training configuration."""
        config_dict = load_config(self.config.config_file)
        self.train_config = create_namespace_from_dict(config_dict)
        
        if is_main_process():
            print_config_summary(self.train_config)
    
    def setup_model_and_data(self):
        """Setup model, data, and related components."""
        # Setup graph and noise
        vocab_size = getattr(self.train_config, 'tokens', 33)
        self.graph = graph_lib.Absorbing(vocab_size - 1)  # vocab_size includes absorbing token
        self.noise = noise_lib.LogLinearNoise()
        
        # Setup model
        self.model = DiscDiffModel(self.train_config).to(self.device)
        
        # Setup DDP if needed
        if self.config.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.device] if self.device.type == 'cuda' else None)
        
        # Setup data
        self.setup_data_loaders()
        
        print(f"🏗️  Model setup: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"📊 Data setup: {len(self.train_loader)} batches per epoch")
    
    def setup_data_loaders(self):
        """Setup training and validation data loaders."""
        # Training dataset
        train_dataset = UniRef50Dataset(
            data_file=self.config.datafile,
            tokenize_on_fly=getattr(self.train_config.data, 'tokenize_on_fly', False),
            max_length=getattr(self.train_config.data, 'max_length', 512),
            use_streaming=getattr(self.train_config.data, 'use_streaming', False)
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
        
        # Training data loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=getattr(self.train_config.training, 'batch_size', 32),
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=getattr(self.train_config.data, 'num_workers', 4),
            pin_memory=getattr(self.train_config.data, 'pin_memory', True),
            drop_last=True
        )
        
        self.train_sampler = train_sampler
    
    def setup_training_components(self):
        """Setup optimizer, scheduler, EMA, and metrics."""
        # Optimizer
        self.optimizer = create_optimizer(
            self.model,
            learning_rate=float(getattr(self.train_config.training, 'learning_rate', 1e-4)),
            weight_decay=float(getattr(self.train_config.training, 'weight_decay', 0.01))
        )
        
        # Scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            warmup_steps=getattr(self.train_config.training, 'warmup_steps', 1000),
            max_steps=getattr(self.train_config.training, 'max_steps', 100000)
        )
        
        # EMA
        self.ema_model = None
        if getattr(self.train_config.training, 'use_ema', True):
            self.ema_model = EMAModel(
                self.model.module if hasattr(self.model, 'module') else self.model,
                decay=getattr(self.train_config.training, 'ema_decay', 0.9999),
                device=self.device
            )
        
        # Metrics
        self.metrics = TrainingMetrics()
        
        # Training state
        self.current_step = 0
        self.start_time = time.time()
        self.accumulate_grad_batches = getattr(self.train_config.training, 'accumulate_grad_batches', 1)
        self.accumulation_step = 0

        print("🚀 Training components initialized")
        print(f"📊 Gradient accumulation: {self.accumulate_grad_batches} batches")
    
    def train_step(self, batch):
        """Execute a single training step with gradient accumulation support."""
        # Move batch to device
        x0 = batch.to(self.device)
        batch_size = x0.shape[0]

        # Sample time and noise
        t = torch.rand(batch_size, device=self.device)
        sigma = self.noise.sigma(t)

        # Corrupt data
        xt = self.graph.sample_transition(x0, sigma)

        # Forward pass
        if getattr(self.train_config.training, 'use_subs_loss', True):
            # SUBS loss
            model_output = self.model(xt, sigma, use_subs=True)

            # Compute SUBS loss with curriculum learning
            loss, curriculum_info = subs_loss_with_curriculum(
                model_output=model_output,
                x0=x0,
                sigma=sigma,
                noise_schedule=self.noise,
                training_step=self.current_step,
                preschool_time=getattr(self.train_config.curriculum, 'decay_steps', 5000) if hasattr(self.train_config, 'curriculum') else 5000
            )
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
        
        # Setup progress bar
        if is_main_process():
            pbar = tqdm(self.train_loader, desc=f"Training")
        else:
            pbar = self.train_loader
        
        for batch in pbar:
            step_start_time = time.time()

            # Training step
            loss, accuracy, perplexity = self.train_step(batch)

            # Increment accumulation step
            self.accumulation_step += 1

            # Only perform optimization step when we've accumulated enough gradients
            if self.accumulation_step >= self.accumulate_grad_batches:
                # Gradient clipping and optimization
                grad_norm = clip_gradients(
                    self.model,
                    max_norm=getattr(self.train_config.training, 'gradient_clip_norm', 1.0)
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Update EMA
                if self.ema_model is not None:
                    self.ema_model.update(self.model.module if hasattr(self.model, 'module') else self.model)

                # Reset accumulation counter
                self.accumulation_step = 0

                # Increment step counter (only after actual optimization step)
                self.current_step += 1
            else:
                # No optimization step, set grad_norm to 0 for logging
                grad_norm = 0.0
            
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
            if is_main_process():
                pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'acc': f"{accuracy:.3f}",
                    'ppl': f"{perplexity:.2f}",
                    'lr': f"{current_lr:.2e}",
                    'step': self.current_step,
                    'acc_step': f"{self.accumulation_step}/{self.accumulate_grad_batches}"
                })

            # Log metrics periodically (only after actual optimization steps)
            if self.accumulation_step == 0 and self.current_step % 100 == 0 and is_main_process():
                self.log_training_metrics()

            # Save checkpoint periodically (only after actual optimization steps)
            if self.accumulation_step == 0 and self.current_step % 5000 == 0 and is_main_process():
                self.save_training_checkpoint()
        
        return epoch_metrics.get_averages()
    
    def log_training_metrics(self):
        """Log training metrics."""
        metrics = self.metrics.get_averages(window=100)
        
        # Add timing information
        elapsed_time = time.time() - self.start_time
        metrics['elapsed_time'] = elapsed_time
        metrics['steps_per_second'] = self.current_step / elapsed_time
        
        # Log to wandb if available
        if hasattr(self, 'wandb_run') and self.wandb_run is not None:
            log_metrics(metrics, self.current_step, wandb_run=self.wandb_run)
        
        # Print summary
        print(f"\n📊 Step {self.current_step} | "
              f"Loss: {metrics['loss']:.4f} | "
              f"Acc: {metrics['accuracy']:.3f} | "
              f"PPL: {metrics['perplexity']:.2f} | "
              f"LR: {metrics['learning_rate']:.2e} | "
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
        # Setup wandb
        if self.config.use_wandb and is_main_process():
            self.wandb_run = setup_wandb(wandb_project, wandb_name, self.train_config.__dict__)
        else:
            self.wandb_run = None
        
        print(f"\n🚀 Starting training...")
        print(f"📊 Total steps: {getattr(self.train_config.training, 'max_steps', 100000)}")
        
        try:
            # Training loop
            max_steps = getattr(self.train_config.training, 'max_steps', 100000)
            
            while self.current_step < max_steps:
                # Set epoch for distributed sampler
                if self.train_sampler is not None:
                    epoch = self.current_step // len(self.train_loader)
                    self.train_sampler.set_epoch(epoch)
                
                # Train epoch
                epoch_metrics = self.train_epoch()
                
                if is_main_process():
                    print(f"\n✅ Epoch completed | Avg Loss: {epoch_metrics['loss']:.4f}")
                
                # Check if we've reached max steps
                if self.current_step >= max_steps:
                    break
            
            # Final checkpoint
            if is_main_process():
                self.save_training_checkpoint()
                print(f"\n🎉 Training completed! Final step: {self.current_step}")
        
        finally:
            # Cleanup
            if self.wandb_run is not None:
                self.wandb_run.finish()
            
            if self.config.world_size > 1:
                cleanup_ddp()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clean UniRef50 Discrete Diffusion Training")
    
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--datafile", type=str, required=True, help="Path to training data")
    parser.add_argument("--work_dir", type=str, default="./work_dir", help="Working directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--resume_checkpoint", type=str, help="Resume from checkpoint")
    parser.add_argument("--wandb_project", type=str, default="sedd-training", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, help="Wandb run name")
    
    # DDP arguments
    parser.add_argument("--cluster", type=str, choices=["aurora", "polaris"], help="Cluster type for DDP")
    
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
