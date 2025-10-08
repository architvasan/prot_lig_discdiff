"""
Rotary positional embeddings implementation
Adapted from the main MDLM codebase for V100 compatibility
"""
import torch
import torch.nn as nn
import math

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
            # This makes the transformation on v an identity.
            self.cos_cached[:,:,2,:,:].fill_(1.)
            self.sin_cached[:,:,2,:,:].fill_(0.)

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
    """
    V100-compatible rotary position embedding application
    Fallback implementation without flash attention
    """
    # Extract cos and sin for the sequence
    cos = cos[0, :qkv.shape[1], 0, 0, :]  # [seq_len, dim]
    sin = sin[0, :qkv.shape[1], 0, 0, :]  # [seq_len, dim]
    
    # Apply to q and k only (not v)
    q, k, v = qkv.unbind(dim=2)  # Each is [batch, seq, heads, dim]
    
    # Reshape for rotary application
    q_rot = q * cos.unsqueeze(0).unsqueeze(2) + rotate_half(q) * sin.unsqueeze(0).unsqueeze(2)
    k_rot = k * cos.unsqueeze(0).unsqueeze(2) + rotate_half(k) * sin.unsqueeze(0).unsqueeze(2)
    
    # Stack back together
    qkv_rot = torch.stack([q_rot, k_rot, v], dim=2)
    
    return qkv_rot


# Alternative implementation for different input formats
def apply_rotary_pos_emb_simple(q, k, cos, sin):
    """
    Simple rotary embedding for separate q, k tensors
    """
    # Apply rotary embedding
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed
