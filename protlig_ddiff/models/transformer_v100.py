import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

try:
    from . import rotary
except ImportError:
    try:
        from protlig_ddiff.models import rotary
    except ImportError:
        import rotary
# Import fused operations but provide fallbacks for cross-platform compatibility
try:
    from .fused_add_dropout_scale import (
        bias_dropout_add_scale_fused_train,
        bias_dropout_add_scale_fused_inference,
        get_bias_dropout_add_scale,
        modulate_fused,
    )
    FUSED_OPS_AVAILABLE = True
except:
    FUSED_OPS_AVAILABLE = False


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def device_compatible_modulate(x, shift, scale):
    """Device-compatible modulation without JIT compilation."""
    # shift and scale already have the right shape for broadcasting, don't unsqueeze
    return x * (1 + scale) + shift

def device_compatible_bias_dropout_scale(x, bias, scale, residual, dropout_prob, training=True):
    """Device-compatible bias dropout scale without JIT compilation."""
    if bias is not None:
        out = scale * F.dropout(x + bias, p=dropout_prob, training=training)
    else:
        out = scale * F.dropout(x, p=dropout_prob, training=training)

    if residual is not None:
        # Only fix simple shape mismatches
        if out.shape != residual.shape:
            if out.numel() == residual.numel():
                out = out.view(residual.shape)
            else:
                # If shapes are incompatible, skip residual (this shouldn't happen in working model)
                print(f"WARNING: Incompatible residual shapes {out.shape} vs {residual.shape}, skipping")
                return out
        out = out + residual
    return out


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        # Cross-platform layer norm
        device_type = str(x.device).split(':')[0]
        if device_type == 'cuda':
            with torch.cuda.amp.autocast(enabled=False):
                x = F.layer_norm(x.float(), [self.dim])
        else:
            # For CPU and MPS, use standard layer norm
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


class FusedMLP(nn.Module):
    """V100-compatible MLP replacement for flash_attn FusedMLP"""
    def __init__(self, dim, hidden_dim, activation='gelu'):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.activation = getattr(F, activation)
    
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class FusedDense(nn.Module):
    """V100-compatible dense layer replacement"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # Flatten t to handle different input shapes
        original_shape = t.shape
        t_flat = t.view(-1)

        t_freq = self.timestep_embedding(t_flat, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)

        # Reshape back to match original batch dimensions
        if len(original_shape) > 1:
            t_emb = t_emb.view(original_shape[0], -1)

        return t_emb


class V100MultiHeadAttention(nn.Module):
    """V100-compatible multi-head attention with optimizations"""
    
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, cu_seqlens, max_seqlen):
        """
        V100-compatible attention that mimics flash_attn_varlen_qkvpacked_func behavior
        """
        batch_size = cu_seqlens.shape[0] - 1
        total_len = x.shape[0]
        
        # Get QKV
        qkv = self.qkv(x)  # [total_len, 3 * dim]
        qkv = qkv.view(total_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)  # Each: [total_len, n_heads, head_dim]
        
        # Process each sequence in the batch separately
        outputs = []
        for i in range(batch_size):
            start_idx = cu_seqlens[i].item()
            end_idx = cu_seqlens[i + 1].item()
            seq_len = end_idx - start_idx
            
            if seq_len == 0:
                continue
                
            # Extract sequence
            q_seq = q[start_idx:end_idx]  # [seq_len, n_heads, head_dim]
            k_seq = k[start_idx:end_idx]
            v_seq = v[start_idx:end_idx]
            
            # Reshape for attention
            q_seq = q_seq.transpose(0, 1)  # [n_heads, seq_len, head_dim]
            k_seq = k_seq.transpose(0, 1)
            v_seq = v_seq.transpose(0, 1)
            
            # Compute attention
            attn_scores = torch.matmul(q_seq, k_seq.transpose(-2, -1)) * self.scale
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            
            attn_out = torch.matmul(attn_probs, v_seq)  # [n_heads, seq_len, head_dim]
            attn_out = attn_out.transpose(0, 1)  # [seq_len, n_heads, head_dim]
            
            outputs.append(attn_out)
        
        # Concatenate all sequences
        if outputs:
            x = torch.cat(outputs, dim=0)  # [total_len, n_heads, head_dim]
            x = x.contiguous().view(total_len, -1)  # [total_len, dim]
        else:
            x = torch.zeros_like(x)
        
        return self.out_proj(x)


def memory_efficient_attention(q, k, v, chunk_size=512):
    """
    Memory-efficient attention using chunked computation.

    Args:
        q: [n_heads, seq_len, head_dim]
        k: [n_heads, seq_len, head_dim]
        v: [n_heads, seq_len, head_dim]
        chunk_size: Size of chunks to process at once

    Returns:
        output: [n_heads, seq_len, head_dim]
    """
    n_heads, seq_len, head_dim = q.shape
    scale = head_dim ** -0.5

    # If sequence is small enough, use standard attention
    if seq_len <= chunk_size:
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, v)

    # Chunked attention for large sequences
    output = torch.zeros_like(q)

    for i in range(0, seq_len, chunk_size):
        end_i = min(i + chunk_size, seq_len)
        q_chunk = q[:, i:end_i]  # [n_heads, chunk_size, head_dim]

        # Compute attention scores for this query chunk against all keys
        attn_scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale  # [n_heads, chunk_size, seq_len]
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        output[:, i:end_i] = torch.matmul(attn_weights, v)  # [n_heads, chunk_size, head_dim]

    return output


def v100_flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, causal=False):
    """
    V100-compatible replacement for flash_attn_varlen_qkvpacked_func
    """
    # qkv shape: [total_len, 3, n_heads, head_dim]
    total_len, three, n_heads, head_dim = qkv.shape
    batch_size = cu_seqlens.shape[0] - 1

    q, k, v = qkv.unbind(dim=1)  # Each: [total_len, n_heads, head_dim]
    scale = head_dim ** -0.5

    outputs = []
    for i in range(batch_size):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()
        seq_len = end_idx - start_idx

        if seq_len == 0:
            continue

        # Extract sequence
        q_seq = q[start_idx:end_idx].transpose(0, 1)  # [n_heads, seq_len, head_dim]
        k_seq = k[start_idx:end_idx].transpose(0, 1)
        v_seq = v[start_idx:end_idx].transpose(0, 1)

        # Use memory-efficient attention for long sequences
        if seq_len > 1024:  # Use chunked attention for long sequences
            chunk_size = min(512, max(64, seq_len // 8))  # Adaptive chunk size
            attn_out = memory_efficient_attention(q_seq, k_seq, v_seq, chunk_size=chunk_size)

            # Apply dropout if needed
            if dropout_p > 0.0:
                attn_out = F.dropout(attn_out, p=dropout_p, training=True)
        else:
            # Standard attention for shorter sequences
            attn_scores = torch.matmul(q_seq, k_seq.transpose(-2, -1)) * scale

            if causal:
                # Apply causal mask
                mask = torch.triu(torch.ones(seq_len, seq_len, device=qkv.device), diagonal=1)
                attn_scores.masked_fill_(mask.bool(), float('-inf'))

            attn_probs = F.softmax(attn_scores, dim=-1)

            if dropout_p > 0.0:
                attn_probs = F.dropout(attn_probs, p=dropout_p, training=True)

            attn_out = torch.matmul(attn_probs, v_seq)  # [n_heads, seq_len, head_dim]

        attn_out = attn_out.transpose(0, 1)  # [seq_len, n_heads, head_dim]

        outputs.append(attn_out)

    if outputs:
        result = torch.cat(outputs, dim=0)  # [total_len, n_heads, head_dim]
        return result  # Return with proper shape for rearrange
    else:
        return torch.zeros(total_len, n_heads, head_dim, device=qkv.device, dtype=qkv.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = FusedMLP(dim, mlp_dim)
        
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        """Get device-compatible bias dropout scale function."""
        if FUSED_OPS_AVAILABLE:
            try:
                if self.training:
                    return bias_dropout_add_scale_fused_train
                else:
                    return bias_dropout_add_scale_fused_inference
            except:
                # Fallback to device-compatible version
                return lambda x, bias, scale, residual, dropout: device_compatible_bias_dropout_scale(
                    x, bias, scale, residual, dropout, training=self.training
                )
        else:
            return lambda x, bias, scale, residual, dropout: device_compatible_bias_dropout_scale(
                x, bias, scale, residual, dropout, training=self.training
            )

    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # Self-attention
        x_skip = x
        # Use device-compatible modulation
        if FUSED_OPS_AVAILABLE:
            try:
                x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
            except:
                x = device_compatible_modulate(self.norm1(x), shift_msa, scale_msa)
        else:
            x = device_compatible_modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.attn_qkv(x)

        # Ensure qkv has the correct shape [batch, seq, 3*dim]
        expected_qkv_shape = (batch_size, seq_len, 3 * self.dim)
        if qkv.shape != expected_qkv_shape:
            print(f"WARNING: qkv shape {qkv.shape} != expected {expected_qkv_shape}")
            # Force reshape to expected shape
            qkv = qkv.view(expected_qkv_shape)
            print(f"Fixed qkv shape to: {qkv.shape}")

        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        
        # Apply rotary embeddings with device compatibility
        device_type = str(qkv.device).split(':')[0]
        if device_type == 'cuda':
            with torch.cuda.amp.autocast(enabled=False):
                cos, sin = rotary_cos_sin
                qkv = rotary.apply_rotary_pos_emb(
                    qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
                )
        else:
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        if seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)

        # Use V100-compatible attention
        x = v100_flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0., causal=False)
        
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # mlp operation
        # Use device-compatible modulation
        if FUSED_OPS_AVAILABLE:
            try:
                mlp_input = modulate_fused(self.norm2(x), shift_mlp, scale_mlp)
            except:
                mlp_input = device_compatible_modulate(self.norm2(x), shift_mlp, scale_mlp)
        else:
            mlp_input = device_compatible_modulate(self.norm2(x), shift_mlp, scale_mlp)

        x = bias_dropout_scale_fn(self.mlp(mlp_input), None, gate_mlp, x, self.dropout)
        return x


class OutputLayer(nn.Module):
    def __init__(self, dim, vocab_size, cond_dim):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.linear = nn.Linear(dim, vocab_size, bias=False)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        # Use device-compatible modulation
        if FUSED_OPS_AVAILABLE:
            try:
                x = modulate_fused(self.norm(x), shift, scale)
            except:
                x = device_compatible_modulate(self.norm(x), shift, scale)
        else:
            x = device_compatible_modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class DiscDiffModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        self.config = config

        # Handle absorbing state (similar to original transformer.py)
        if hasattr(config, 'graph') and hasattr(config.graph, 'type'):
            self.absorb = config.graph.type == "absorb"
        else:
            self.absorb = True

        # Handle vocab size with absorbing state
        if hasattr(config, 'tokens'):
            base_vocab_size = config.tokens
        elif hasattr(config, 'data') and hasattr(config.data, 'vocab_size_protein'):
            base_vocab_size = config.data.vocab_size_protein
        elif hasattr(config, 'model') and hasattr(config.model, 'vocab_size'):
            base_vocab_size = config.model.vocab_size
        else:
            base_vocab_size = 33  # Default protein vocab size

        # For absorbing state, the vocab_size already includes the absorbing token
        # Don't add 1 because the graph already accounts for it
        vocab_size = base_vocab_size
        self.vocab_size = vocab_size

        # Handle scale_by_sigma setting
        if hasattr(config, 'model'):
            self.scale_by_sigma = getattr(config.model, 'scale_by_sigma', False)
        else:
            self.scale_by_sigma = False

        # Handle model dimensions
        if hasattr(config, 'model'):
            dim = getattr(config.model, 'dim', getattr(config.model, 'hidden_size', 768))
            n_layers = getattr(config.model, 'n_layers', getattr(config.model, 'n_blocks_prot', 8))
            n_heads = getattr(config.model, 'n_heads', 12)
            cond_dim = getattr(config.model, 'cond_dim', 256)
        else:
            dim = 768
            n_layers = 8
            n_heads = 12
            cond_dim = 256

        self.vocab_embed = nn.Embedding(vocab_size, dim)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = rotary.Rotary(dim // n_heads)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, cond_dim) for _ in range(n_layers)
        ])
        
        self.output_layer = OutputLayer(dim, self.vocab_size, cond_dim)

    def _subs_parameterization(self, logits, xt):
        """SUBS parameterization for MDLM loss"""
        # Get mask token index (absorbing state)
        mask_index = self.vocab_size - 1  # Your absorbing state token
        neg_infinity = -1000000.0

        # Step 1: Set log prob at mask index to -infinity (mask token should never be predicted)
        logits[:, :, mask_index] = neg_infinity

        # Step 2: Normalize logits to log probabilities
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # Step 3: For unmasked positions ONLY, set all probs to -inf except the current token
        unmasked_indices = (xt != mask_index)
        if unmasked_indices.any():
            # Set all tokens to -inf for unmasked positions
            logits[unmasked_indices] = neg_infinity

            # Set current token log prob to 0 for unmasked positions
            unmasked_tokens = xt[unmasked_indices]  # [num_unmasked]
            batch_indices, seq_indices = torch.where(unmasked_indices)

            # Use explicit loop to set the correct positions to 0
            for i, (b_idx, s_idx) in enumerate(zip(batch_indices, seq_indices)):
                logits[b_idx, s_idx, unmasked_tokens[i]] = 0

        # Step 4: For masked positions, keep the normalized distribution over non-mask tokens
        # The mask token should already have -inf log prob from step 1+2, and the distribution
        # should be over non-mask tokens, which is what we want for the loss computation

        return logits

    def forward(self, indices, sigma, use_subs=False):
        # Ensure indices is 2D: [batch_size, sequence_length]
        if indices.dim() != 2:
            print(f"WARNING: Model received {indices.dim()}D indices with shape {indices.shape}")
            if indices.dim() > 2:
                indices = indices.view(indices.shape[0], -1)
                print(f"Reshaped indices to: {indices.shape}")
            else:
                raise ValueError(f"indices must be 2D, got {indices.dim()}D with shape {indices.shape}")

        x = self.vocab_embed(indices)

        # Simple shape validation without aggressive fixing
        if x.dim() != 3:
            print(f"WARNING: vocab_embed output has unexpected shape {x.shape}, expected 3D")
            # Only fix if it's a simple case
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            elif x.dim() > 3:
                # Flatten to 3D conservatively
                x = x.view(indices.shape[0], indices.shape[1], -1)

        c = self.sigma_map(sigma)  # TimestepEmbedder already includes SiLU

        rotary_cos_sin = self.rotary_emb(x)

        # Use appropriate autocast based on device
        device_type = str(x.device).split(':')[0]
        if device_type == 'cuda':
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for i in range(len(self.blocks)):
                    x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
                x = self.output_layer(x, c)
        elif device_type == 'mps':
            # MPS has compatibility issues with some operations, use standard computation
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
            x = self.output_layer(x, c)
        else:
            # CPU - no autocast needed
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
            x = self.output_layer(x, c)

        # Apply SUBS parameterization if requested
        if use_subs:
            x = self._subs_parameterization(x, indices)
            return x

        # Original score-based parameterization
        if self.scale_by_sigma:
            assert self.absorb, "Haven't configured this to work."
            # Ensure sigma has the right shape for broadcasting
            sigma_reshaped = sigma.view(sigma.shape[0], 1, 1)  # [batch, 1, 1]
            esigm1_log = torch.where(sigma_reshaped < 0.5, torch.expm1(sigma_reshaped), sigma_reshaped.exp() - 1).log().to(x.dtype)
            x = x - esigm1_log - np.log(x.shape[-1] - 1)

        # Handle absorbing state (zero out positions that match input indices)
        if self.absorb:
             # Ensure tensors are on the same device and have compatible shapes
             with torch.cuda.amp.autocast(enabled=False):
                 # Check shape compatibility before scatter
                 if x.shape[:2] == indices.shape:
                     indices_expanded = indices.unsqueeze(-1)  # [batch, seq, 1]
                     zeros = torch.zeros_like(x[..., :1])  # [batch, seq, 1]
                     x = torch.scatter(x, -1, indices_expanded, zeros)
                 else:
                     print(f"WARNING: Skipping absorbing state scatter due to shape mismatch:")
                     print(f"  x shape: {x.shape}, indices shape: {indices.shape}")
                     # Try to fix the shape if possible
                     if x.numel() == indices.shape[0] * indices.shape[1] * x.shape[-1]:
                         x = x.view(indices.shape[0], indices.shape[1], -1)
                         indices_expanded = indices.unsqueeze(-1)
                         zeros = torch.zeros_like(x[..., :1])
                         x = torch.scatter(x, -1, indices_expanded, zeros)
                     else:
                         print(f"  Cannot fix shape mismatch, skipping scatter operation")

        return x
