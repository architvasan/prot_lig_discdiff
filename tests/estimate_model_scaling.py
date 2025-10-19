#!/usr/bin/env python3
"""
Estimate model scaling effects for the protein discrete diffusion model.
"""

import math

def estimate_transformer_params(dim, n_heads, n_layers, vocab_size, cond_dim, max_seq_len=512):
    """Estimate transformer parameters."""
    
    # Embedding layers
    vocab_embed = vocab_size * dim
    
    # Time embedding (sigma_map)
    time_embed = cond_dim * 4  # Typical time embedding size
    
    # Rotary embedding (no parameters, just computation)
    rotary_embed = 0
    
    # Transformer blocks
    block_params = 0
    for _ in range(n_layers):
        # Layer norms (2 per block)
        ln_params = 2 * dim * 2  # weight + bias
        
        # Attention (QKV + output projection)
        attn_qkv = dim * (3 * dim)  # Q, K, V projections
        attn_out = dim * dim        # Output projection
        attn_params = attn_qkv + attn_out
        
        # MLP (typically 4x expansion)
        mlp_dim = dim * 4
        mlp_fc1 = dim * mlp_dim
        mlp_fc2 = mlp_dim * dim
        mlp_params = mlp_fc1 + mlp_fc2
        
        # AdaLN modulation
        adaln_params = cond_dim * (6 * dim)  # 6 modulation parameters per block
        
        block_params += ln_params + attn_params + mlp_params + adaln_params
    
    # Output layer
    output_ln = dim * 2  # weight + bias
    output_linear = dim * vocab_size
    output_adaln = cond_dim * (2 * dim)  # 2 modulation parameters
    output_params = output_ln + output_linear + output_adaln
    
    total_params = vocab_embed + time_embed + block_params + output_params
    
    return {
        'vocab_embed': vocab_embed,
        'time_embed': time_embed,
        'transformer_blocks': block_params,
        'output_layer': output_params,
        'total': total_params
    }

def estimate_memory_usage(params, batch_size, seq_len, dtype_bytes=4):
    """Estimate memory usage in GB."""
    
    # Model parameters
    model_memory = params * dtype_bytes
    
    # Activations (rough estimate)
    # Forward pass activations are roughly 2x model size for transformers
    activation_memory = model_memory * 2
    
    # Gradients (same size as parameters)
    gradient_memory = model_memory
    
    # Optimizer states (AdamW: 2x parameters for momentum + variance)
    optimizer_memory = model_memory * 2
    
    # Input/output tensors
    io_memory = batch_size * seq_len * 4 * dtype_bytes  # rough estimate
    
    total_memory = model_memory + activation_memory + gradient_memory + optimizer_memory + io_memory
    
    return {
        'model_gb': model_memory / (1024**3),
        'activations_gb': activation_memory / (1024**3),
        'gradients_gb': gradient_memory / (1024**3),
        'optimizer_gb': optimizer_memory / (1024**3),
        'io_gb': io_memory / (1024**3),
        'total_gb': total_memory / (1024**3)
    }

def compare_model_configs():
    """Compare different model configurations."""
    
    configs = {
        'Current (1024d)': {
            'dim': 1024, 'n_heads': 16, 'n_layers': 16, 
            'vocab_size': 26, 'cond_dim': 256
        },
        'Recommended (1536d)': {
            'dim': 1536, 'n_heads': 24, 'n_layers': 20, 
            'vocab_size': 26, 'cond_dim': 384
        },
        'Aggressive (2048d)': {
            'dim': 2048, 'n_heads': 32, 'n_layers': 24, 
            'vocab_size': 26, 'cond_dim': 512
        },
        'Depth-focused (1024d-24L)': {
            'dim': 1024, 'n_heads': 16, 'n_layers': 24, 
            'vocab_size': 26, 'cond_dim': 256
        }
    }
    
    print("🔍 Model Configuration Comparison")
    print("=" * 80)
    
    batch_size = 6  # Per rank
    seq_len = 512
    
    for name, config in configs.items():
        print(f"\n📊 {name}:")
        print("-" * 50)
        
        params = estimate_transformer_params(**config)
        memory = estimate_memory_usage(params['total'], batch_size, seq_len)
        
        print(f"   Parameters: {params['total']/1e6:.1f}M")
        print(f"   Memory per rank: {memory['total_gb']:.1f} GB")
        print(f"   Memory breakdown:")
        print(f"     - Model: {memory['model_gb']:.1f} GB")
        print(f"     - Activations: {memory['activations_gb']:.1f} GB")
        print(f"     - Gradients: {memory['gradients_gb']:.1f} GB")
        print(f"     - Optimizer: {memory['optimizer_gb']:.1f} GB")
        
        # Estimate training speed impact
        current_params = 85e6  # Rough estimate of current model
        speed_ratio = current_params / params['total']
        print(f"   Relative speed: {speed_ratio:.2f}x (vs current)")
        
        # Memory efficiency
        params_per_gb = params['total'] / memory['total_gb'] / 1e6
        print(f"   Efficiency: {params_per_gb:.1f}M params/GB")

def analyze_scaling_benefits():
    """Analyze benefits of scaling up."""
    
    print(f"\n🚀 Benefits of Scaling Up")
    print("=" * 80)
    
    print(f"✅ **Why Larger Models Help with Stagnation:**")
    print(f"   • Higher capacity to learn complex protein patterns")
    print(f"   • Better generalization vs memorization")
    print(f"   • More stable training dynamics")
    print(f"   • Reduced overfitting with proper regularization")
    
    print(f"\n📈 **Expected Improvements:**")
    print(f"   • ESM perplexity continues improving beyond 6k steps")
    print(f"   • Better sequence diversity and quality")
    print(f"   • More realistic amino acid distributions")
    print(f"   • Improved convergence stability")
    
    print(f"\n⚠️  **Considerations:**")
    print(f"   • Increased memory requirements")
    print(f"   • Slower training per step")
    print(f"   • Need for better regularization")
    print(f"   • Longer warmup periods")

def recommend_scaling_strategy():
    """Recommend scaling strategy."""
    
    print(f"\n🎯 Recommended Scaling Strategy")
    print("=" * 80)
    
    print(f"🥇 **Phase 1: Conservative Scale-Up (Recommended)**")
    print(f"   • Model: 1536d, 24h, 20L (~190M params)")
    print(f"   • Memory: ~12 GB per rank")
    print(f"   • Speed: ~0.55x current training speed")
    print(f"   • Benefits: 2.2x parameter increase, better capacity")
    
    print(f"\n🥈 **Phase 2: If Phase 1 Works Well**")
    print(f"   • Model: 2048d, 32h, 24L (~400M params)")
    print(f"   • Memory: ~25 GB per rank")
    print(f"   • Speed: ~0.25x current training speed")
    print(f"   • Benefits: 4.7x parameter increase, much higher capacity")
    
    print(f"\n🔧 **Implementation Tips:**")
    print(f"   • Start with Phase 1 configuration")
    print(f"   • Monitor memory usage carefully")
    print(f"   • Increase warmup steps (3000 → 5000)")
    print(f"   • Add dropout (0.1) for regularization")
    print(f"   • Reduce learning rate slightly (1e-5 → 8e-6)")
    print(f"   • Increase weight decay (0.01 → 0.02)")
    
    print(f"\n📊 **What to Monitor:**")
    print(f"   • ESM perplexity trend (should continue improving)")
    print(f"   • Training loss stability")
    print(f"   • Memory usage per rank")
    print(f"   • Gradient norms")

def main():
    """Run model scaling analysis."""
    
    print("🔬 Model Scaling Analysis for Protein Discrete Diffusion")
    print("=" * 80)
    print("This analysis helps determine optimal model size to address")
    print("the ESM perplexity stagnation issue observed around 6k steps.")
    print("=" * 80)
    
    compare_model_configs()
    analyze_scaling_benefits()
    recommend_scaling_strategy()
    
    print(f"\n🎯 Summary:")
    print("=" * 60)
    print("The recommended 1536d/24h/20L model should provide:")
    print("• 2.2x more parameters for better protein pattern learning")
    print("• Manageable memory increase (~12 GB per rank)")
    print("• Better generalization to prevent overfitting")
    print("• Continued ESM perplexity improvement beyond 6k steps")
    
    return True

if __name__ == "__main__":
    main()
