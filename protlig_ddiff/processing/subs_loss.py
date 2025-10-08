"""
SUBS (Substitution) Loss Implementation for MDLM
"""
import torch
import torch.nn.functional as F


def subs_loss(model_output, x0, sigma, noise_schedule, attention_mask=None):
    """
    MDLM SUBS loss computation
    
    Args:
        model_output: Model log probabilities [batch, seq, vocab] 
        x0: Original clean tokens [batch, seq]
        sigma: Noise level [batch]
        noise_schedule: Your noise schedule object
        attention_mask: Optional attention mask [batch, seq]
    
    Returns:
        loss: Scalar loss value
    """
    # Get noise derivatives
    _, dsigma = noise_schedule(sigma.unsqueeze(-1))  # [batch, 1]
    
    # Extract log probabilities of ground truth tokens
    log_p_theta = torch.gather(
        input=model_output,
        dim=-1, 
        index=x0[:, :, None]
    ).squeeze(-1)  # [batch, seq]
    
    # SUBS loss: negative log likelihood weighted by noise derivative
    # Formula: -log p_θ(x_0|x_t) * (dσ/dt / (exp(σ) - 1))
    weight = dsigma / torch.expm1(sigma.unsqueeze(-1))  # [batch, 1]
    loss_per_token = -log_p_theta * weight  # [batch, seq]
    
    # Apply attention mask if provided
    if attention_mask is not None:
        loss_per_token = loss_per_token * attention_mask
        # Normalize by number of valid tokens
        total_loss = loss_per_token.sum()
        total_tokens = attention_mask.sum()
        return total_loss / total_tokens
    else:
        return loss_per_token.mean()


def subs_loss_with_curriculum(model_output, x0, sigma, noise_schedule, 
                             training_step, preschool_time=5000, 
                             attention_mask=None):
    """
    SUBS loss with curriculum learning integration
    
    Args:
        model_output: Model log probabilities [batch, seq, vocab]
        x0: Original clean tokens [batch, seq] 
        sigma: Noise level [batch]
        noise_schedule: Your noise schedule object
        training_step: Current training step
        preschool_time: Steps for curriculum ramp-up
        attention_mask: Optional attention mask [batch, seq]
    
    Returns:
        loss: Scalar loss value
        curriculum_info: Dict with curriculum statistics
    """
    # Calculate curriculum progress
    progress = min(1.0, float(training_step) / float(preschool_time))
    
    # Compute base SUBS loss
    base_loss = subs_loss(model_output, x0, sigma, noise_schedule, attention_mask)
    
    # Optional: Apply curriculum weighting (you can experiment with this)
    # For now, just use the base loss since your curriculum is in timestep sampling
    curriculum_loss = base_loss
    
    # Curriculum statistics for logging
    curriculum_info = {
        'curriculum_progress': progress,
        'base_loss': base_loss.item(),
        'curriculum_loss': curriculum_loss.item(),
        'mean_sigma': sigma.mean().item(),
        'std_sigma': sigma.std().item(),
    }
    
    return curriculum_loss, curriculum_info


class SUBSLossModule(torch.nn.Module):
    """
    PyTorch module wrapper for SUBS loss
    """
    def __init__(self, noise_schedule):
        super().__init__()
        self.noise_schedule = noise_schedule
        
    def forward(self, model_output, x0, sigma, attention_mask=None):
        return subs_loss(model_output, x0, sigma, self.noise_schedule, attention_mask)


def compute_subs_metrics(model_output, x0, sigma, attention_mask=None):
    """
    Compute additional metrics for monitoring SUBS training
    
    Args:
        model_output: Model log probabilities [batch, seq, vocab]
        x0: Original clean tokens [batch, seq]
        sigma: Noise level [batch] 
        attention_mask: Optional attention mask [batch, seq]
    
    Returns:
        metrics: Dict with various metrics
    """
    # Extract log probabilities of ground truth tokens
    log_p_theta = torch.gather(
        input=model_output,
        dim=-1,
        index=x0[:, :, None]
    ).squeeze(-1)  # [batch, seq]
    
    # Convert to probabilities for interpretability
    p_theta = torch.exp(log_p_theta)
    
    # Apply attention mask if provided
    if attention_mask is not None:
        valid_mask = attention_mask.bool()
        log_p_theta_valid = log_p_theta[valid_mask]
        p_theta_valid = p_theta[valid_mask]
    else:
        log_p_theta_valid = log_p_theta.flatten()
        p_theta_valid = p_theta.flatten()
    
    metrics = {
        'mean_log_prob': log_p_theta_valid.mean().item(),
        'std_log_prob': log_p_theta_valid.std().item(),
        'mean_prob': p_theta_valid.mean().item(),
        'min_prob': p_theta_valid.min().item(),
        'max_prob': p_theta_valid.max().item(),
        'perplexity': torch.exp(-log_p_theta_valid.mean()).item(),
        'accuracy': (p_theta_valid > 0.5).float().mean().item(),  # Rough accuracy measure
    }
    
    return metrics
