"""
Training utilities for optimization, scheduling, and monitoring.
"""
import torch
import torch.nn as nn
import numpy as np
import wandb
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json
import time


def create_optimizer(model, learning_rate=1e-4, weight_decay=0.01, optimizer_type='adamw'):
    """Create optimizer for the model."""
    if optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer, warmup_steps=1000, max_steps=100000, scheduler_type='cosine'):
    """Create learning rate scheduler."""
    if scheduler_type.lower() == 'cosine':
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type.lower() == 'linear':
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return max(0.0, (max_steps - step) / (max_steps - warmup_steps))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type.lower() == 'constant':
        def lr_lambda(step):
            return 1.0 if step >= warmup_steps else step / warmup_steps
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return scheduler


class EMAModel:
    """Exponential Moving Average model for stable training."""
    
    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        
        # Create EMA model
        self.ema_model = type(model)(model.config if hasattr(model, 'config') else None)
        self.ema_model.to(self.device)
        self.ema_model.eval()
        
        # Initialize EMA parameters
        self.ema_model.load_state_dict(model.state_dict())
        
        # Disable gradients for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
    
    def update(self, model):
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
    
    def state_dict(self):
        """Get EMA model state dict."""
        return self.ema_model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load EMA model state dict."""
        self.ema_model.load_state_dict(state_dict)


class TrainingMetrics:
    """Track and compute training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.losses = []
        self.accuracies = []
        self.perplexities = []
        self.avg_sigmas = []
        self.learning_rates = []
        self.gradient_norms = []
        self.step_times = []

    def update(self, loss, accuracy=None, perplexity=None, avg_sigma=None, lr=None, grad_norm=None, step_time=None):
        """Update metrics with new values."""
        self.losses.append(loss)
        if accuracy is not None:
            self.accuracies.append(accuracy)
        if perplexity is not None:
            self.perplexities.append(perplexity)
        if avg_sigma is not None:
            self.avg_sigmas.append(avg_sigma)
        if lr is not None:
            self.learning_rates.append(lr)
        if grad_norm is not None:
            self.gradient_norms.append(grad_norm)
        if step_time is not None:
            self.step_times.append(step_time)
    
    def get_averages(self, window=100):
        """Get average metrics over the last window steps."""
        metrics = {}
        
        if self.losses:
            metrics['loss'] = np.mean(self.losses[-window:])
        if self.accuracies:
            metrics['accuracy'] = np.mean(self.accuracies[-window:])
        if self.perplexities:
            metrics['perplexity'] = np.mean(self.perplexities[-window:])
        if self.avg_sigmas:
            metrics['avg_sigma'] = np.mean(self.avg_sigmas[-window:])
        if self.learning_rates:
            metrics['learning_rate'] = self.learning_rates[-1]  # Current LR
        if self.gradient_norms:
            metrics['gradient_norm'] = np.mean(self.gradient_norms[-window:])
        if self.step_times:
            metrics['step_time'] = np.mean(self.step_times[-window:])
            metrics['steps_per_second'] = 1.0 / metrics['step_time'] if metrics['step_time'] > 0 else 0
        
        return metrics


def compute_accuracy(predictions, targets, ignore_index=-100):
    """Compute token-level accuracy."""
    # Get predicted tokens
    pred_tokens = torch.argmax(predictions, dim=-1)
    
    # Create mask for valid tokens
    valid_mask = (targets != ignore_index)
    
    # Compute accuracy
    correct = (pred_tokens == targets) & valid_mask
    accuracy = correct.sum().float() / valid_mask.sum().float()
    
    return accuracy.item()


def compute_perplexity(model_output, targets, ignore_index=-100):
    """
    Compute perplexity from model outputs and targets.

    Args:
        model_output: Model logits [batch, seq, vocab] or log probabilities
        targets: Target tokens [batch, seq]
        ignore_index: Index to ignore (padding, special tokens)

    Returns:
        perplexity: Perplexity value
    """
    # Flatten for easier processing
    flat_output = model_output.view(-1, model_output.size(-1))  # [batch*seq, vocab]
    flat_targets = targets.view(-1)  # [batch*seq]

    # Create mask for valid tokens (not padding/special tokens)
    valid_mask = (flat_targets != ignore_index)

    if valid_mask.sum() == 0:
        return float('inf')

    # Calculate cross-entropy loss (negative log-likelihood) for valid tokens only
    # Use reduction='none' to get per-token losses
    if model_output.dim() == 3:  # Logits
        log_probs = F.log_softmax(flat_output, dim=-1)
    else:  # Already log probabilities
        log_probs = flat_output

    # Extract log probabilities for target tokens
    target_log_probs = log_probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)

    # Apply mask and calculate average negative log-likelihood
    valid_log_probs = target_log_probs[valid_mask]
    avg_nll = -valid_log_probs.mean()

    # Perplexity is exp(average negative log-likelihood)
    perplexity = torch.exp(avg_nll)

    return perplexity.item()


def clip_gradients(model, max_norm=1.0):
    """Clip gradients and return the gradient norm."""
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return grad_norm.item()


def save_checkpoint(model, optimizer, scheduler, ema_model, step, loss, save_path):
    """Save training checkpoint."""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    
    if ema_model is not None:
        checkpoint['ema_state_dict'] = ema_model.state_dict()
    
    torch.save(checkpoint, save_path)
    # print(f"✅ Checkpoint saved: {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, ema_model=None):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load EMA state
    if ema_model is not None and 'ema_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
    
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    # print(f"✅ Checkpoint loaded: step {step}, loss {loss:.4f}")
    return step, loss


def setup_wandb(project_name, run_name, config, resume_id=None):
    """Setup Weights & Biases logging."""
    try:
        if resume_id:
            run = wandb.init(
                project=project_name,
                name=run_name,
                config=config,
                resume="must",
                id=resume_id
            )
        else:
            run = wandb.init(
                project=project_name,
                name=run_name,
                config=config
            )
        
        # print(f"✅ Wandb initialized: {project_name}/{run_name}")
        return run

    except Exception as e:
        # print(f"⚠️  Failed to initialize wandb: {e}")
        return None


def log_metrics(metrics, step, wandb_run='protlig_dd', log_file=None):
    """Log metrics to wandb and/or file."""
    # Log to wandb
    if wandb_run is not None:
        try:
            wandb_run.log(metrics, step=step)
        except Exception as e:
            # print(f"⚠️  Failed to log to wandb: {e}")
            pass
    
    # Log to file
    if log_file is not None:
        try:
            log_entry = {'step': step, **metrics, 'timestamp': time.time()}
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            # print(f"⚠️  Failed to log to file: {e}")
            pass


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def estimate_remaining_time(current_step, total_steps, elapsed_time):
    """Estimate remaining training time."""
    if current_step == 0:
        return "Unknown"
    
    steps_per_second = current_step / elapsed_time
    remaining_steps = total_steps - current_step
    remaining_seconds = remaining_steps / steps_per_second
    
    return format_time(remaining_seconds)
