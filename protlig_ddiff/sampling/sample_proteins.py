#!/usr/bin/env python3
"""
Standalone script for sampling protein sequences with conditioning.
"""

import torch
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from protlig_ddiff.sampling.protein_sampling import ProteinSampler
from protlig_ddiff.utils.data_utils import ProteinTokenizer
from protlig_ddiff.models.transformer_v100 import DiscDiffModel
import protlig_ddiff.processing.graph_lib as graph_lib
import protlig_ddiff.processing.noise_lib as noise_lib


def load_model_from_checkpoint(checkpoint_path, config_path, device='cuda'):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        device: Device to load on
        
    Returns:
        Tuple of (model, graph, noise, config)
    """
    
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config namespace
    class ConfigNamespace:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigNamespace(value))
                else:
                    setattr(self, key, value)
    
    config = ConfigNamespace(config_dict)
    
    # Setup graph and noise
    vocab_size = config_dict.get('tokens', 26)
    graph = graph_lib.Absorbing(vocab_size - 1)
    
    noise_type = config_dict.get('noise', {}).get('type', 'loglinear').lower()
    if noise_type == 'cosine':
        noise = noise_lib.CosineNoise()
    else:
        noise = noise_lib.LogLinearNoise()
    
    # Create model
    model = DiscDiffModel(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    return model, graph, noise, config


def main():
    parser = argparse.ArgumentParser(description="Sample protein sequences with conditioning")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    # Sampling arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Number of sequences to sample")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--steps", type=int, default=512, help="Number of sampling steps")
    parser.add_argument("--predictor", type=str, default="analytic", help="Sampling predictor")
    
    # Conditioning arguments
    parser.add_argument("--prefix", type=str, default="", help="Sequence prefix (e.g., 'M' for methionine start)")
    parser.add_argument("--suffix", type=str, default="", help="Sequence suffix")
    parser.add_argument("--unconditional", action="store_true", help="Sample unconditional sequences")
    
    # Output arguments
    parser.add_argument("--output", type=str, help="Output file to save sequences")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("🧬 Protein Sequence Sampling")
    print("=" * 50)
    
    # Load model
    print(f"📂 Loading model from {args.checkpoint}")
    try:
        model, graph, noise, config = load_model_from_checkpoint(
            args.checkpoint, args.config, args.device
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create sampler
    print("🔧 Creating protein sampler...")
    tokenizer = ProteinTokenizer()
    sampler = ProteinSampler(model, graph, noise, tokenizer, args.device)
    
    # Sample sequences
    print(f"🎲 Sampling {args.batch_size} sequences...")
    print(f"   Max length: {args.max_length}")
    print(f"   Steps: {args.steps}")
    print(f"   Predictor: {args.predictor}")
    
    if args.prefix or args.suffix:
        print(f"   Prefix: '{args.prefix}'")
        print(f"   Suffix: '{args.suffix}'")
    
    try:
        if args.unconditional:
            sequences = sampler.sample_unconditional(
                batch_size=args.batch_size,
                max_length=args.max_length,
                steps=args.steps,
                predictor=args.predictor
            )
        else:
            sequences = sampler.sample_with_prefix_suffix(
                prefix=args.prefix,
                suffix=args.suffix,
                batch_size=args.batch_size,
                max_length=args.max_length,
                steps=args.steps,
                predictor=args.predictor
            )
        
        print("✅ Sampling completed!")
        
    except Exception as e:
        print(f"❌ Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display results
    print(f"\n🧬 Generated {len(sequences)} protein sequences:")
    print("=" * 80)
    
    for i, sequence in enumerate(sequences):
        # Clean up sequence
        clean_seq = sequence.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
        
        print(f"Sequence {i+1}:")
        print(f"  Length: {len(clean_seq)}")
        print(f"  Sequence: {clean_seq}")
        
        if args.verbose:
            # Show amino acid composition
            aa_counts = {}
            for aa in clean_seq:
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
            print(f"  Composition: {aa_counts}")
        
        print()
    
    # Save to file if requested
    if args.output:
        print(f"💾 Saving sequences to {args.output}")
        with open(args.output, 'w') as f:
            for i, sequence in enumerate(sequences):
                clean_seq = sequence.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
                f.write(f">sequence_{i+1}\n")
                f.write(f"{clean_seq}\n")
        print("✅ Sequences saved!")


if __name__ == "__main__":
    main()
