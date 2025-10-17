#!/usr/bin/env python3
"""
Test script to demonstrate enhanced time sampling stochasticity modes.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

def test_stochasticity_modes():
    """Test different time sampling stochasticity modes."""
    print("🧪 Testing Enhanced Time Sampling Stochasticity")
    print("=" * 60)
    
    # Mock config for testing
    class MockConfig:
        def __init__(self, mode, **kwargs):
            self.seed = 42
            self.rank = 0
            self.world_size = 4
            self.time_sampling = {
                'mode': mode,
                'extra_entropy_sources': kwargs.get('extra_entropy', False),
                'temporal_mixing': kwargs.get('temporal_mixing', False),
                'sequence_level_noise': kwargs.get('sequence_noise', False),
                'stochasticity_scale': kwargs.get('stoch_scale', 1.0),
            }
    
    # Mock trainer class
    class MockTrainer:
        def __init__(self, config):
            self.config = config
            self.current_step = 1000
            self.device = torch.device('cpu')
        
        def sample_timesteps(self, batch_size, mode='training', batch_idx=None):
            """Copy of the sample_timesteps method for testing."""
            # Get time sampling configuration
            time_config = getattr(self.config, 'time_sampling', {})
            sampling_mode = time_config.get('mode', 'rank_based')
            extra_entropy = time_config.get('extra_entropy_sources', False)
            temporal_mixing = time_config.get('temporal_mixing', False)
            sequence_noise = time_config.get('sequence_level_noise', False)
            stoch_scale = time_config.get('stochasticity_scale', 1.0)
            
            if mode == 'validation':
                # Deterministic validation for reproducibility
                val_generator = torch.Generator(device=self.device).manual_seed(
                    self.config.seed + (batch_idx or 0) * 1000
                )
                return torch.rand(batch_size, device=self.device, generator=val_generator)
            
            elif mode == 'test':
                # Deterministic test for reproducibility
                test_generator = torch.Generator(device=self.device).manual_seed(
                    self.config.seed + (batch_idx or 0) * 2000
                )
                return torch.rand(batch_size, device=self.device, generator=test_generator)
            
            # Training mode - configurable stochasticity
            if sampling_mode == 'deterministic':
                # Fully deterministic (original approach)
                generator = torch.Generator(device=self.device).manual_seed(
                    self.config.seed + self.current_step
                )
                return torch.rand(batch_size, device=self.device, generator=generator)
            
            elif sampling_mode == 'rank_based':
                # Rank-based deterministic (current approach)
                generator = torch.Generator(device=self.device).manual_seed(
                    self.config.seed + self.current_step * self.config.world_size + self.config.rank
                )
                return torch.rand(batch_size, device=self.device, generator=generator)
            
            elif sampling_mode == 'enhanced_stochastic':
                # Enhanced stochasticity with multiple entropy sources
                base_seed = self.config.seed + self.current_step * self.config.world_size + self.config.rank
                
                # Add extra entropy sources
                if extra_entropy:
                    # Add batch index entropy
                    if batch_idx is not None:
                        base_seed += batch_idx * 7919  # Large prime
                    
                    # Add temporal mixing from previous steps
                    if temporal_mixing and self.current_step > 0:
                        prev_step_entropy = (self.current_step - 1) * 6421  # Another prime
                        base_seed += prev_step_entropy
                    
                    # Add some hardware-based entropy (but keep it reproducible)
                    hardware_entropy = hash(str(self.device)) % 10007  # Prime modulo
                    base_seed += hardware_entropy
                
                generator = torch.Generator(device=self.device).manual_seed(base_seed)
                t_base = torch.rand(batch_size, device=self.device, generator=generator)
                
                # Add sequence-level noise within batch
                if sequence_noise:
                    # Generate per-sequence noise
                    seq_generator = torch.Generator(device=self.device).manual_seed(
                        base_seed + 12347  # Different prime offset
                    )
                    seq_noise = torch.rand(batch_size, device=self.device, generator=seq_generator)
                    seq_noise = (seq_noise - 0.5) * 0.1 * stoch_scale  # Small perturbation
                    t_base = torch.clamp(t_base + seq_noise, 0.0, 1.0)
                
                # Apply stochasticity scaling
                if stoch_scale != 1.0:
                    # Scale the variance around 0.5
                    t_centered = t_base - 0.5
                    t_scaled = t_centered * stoch_scale
                    t_base = torch.clamp(t_scaled + 0.5, 0.0, 1.0)
                
                return t_base
            
            elif sampling_mode == 'full_random':
                # Maximum stochasticity - use system randomness
                # Note: This breaks reproducibility but maximizes exploration
                import time
                import os
                
                # Combine multiple entropy sources
                entropy_sources = [
                    self.config.seed,
                    self.current_step,
                    self.config.rank,
                    int(time.time() * 1000000) % 1000000,  # Microsecond timestamp
                    os.getpid() % 10000,  # Process ID
                ]
                
                if batch_idx is not None:
                    entropy_sources.append(batch_idx)
                
                # Mix entropy sources
                mixed_seed = sum(entropy_sources) % (2**32 - 1)
                
                generator = torch.Generator(device=self.device).manual_seed(mixed_seed)
                t_base = torch.rand(batch_size, device=self.device, generator=generator)
                
                # Add additional per-sequence randomness
                if sequence_noise:
                    for i in range(batch_size):
                        seq_seed = (mixed_seed + i * 9973) % (2**32 - 1)  # Prime multiplier
                        seq_gen = torch.Generator(device=self.device).manual_seed(seq_seed)
                        noise = torch.rand(1, device=self.device, generator=seq_gen).item()
                        noise = (noise - 0.5) * 0.2 * stoch_scale  # Larger perturbation
                        t_base[i] = torch.clamp(t_base[i] + noise, 0.0, 1.0)
                
                return t_base
            
            else:
                raise ValueError(f"Unknown time sampling mode: {sampling_mode}")
    
    # Test configurations
    test_configs = [
        ('deterministic', {}),
        ('rank_based', {}),
        ('enhanced_stochastic', {'extra_entropy': False}),
        ('enhanced_stochastic', {'extra_entropy': True}),
        ('enhanced_stochastic', {'extra_entropy': True, 'temporal_mixing': True}),
        ('enhanced_stochastic', {'extra_entropy': True, 'temporal_mixing': True, 'sequence_noise': True}),
        ('enhanced_stochastic', {'extra_entropy': True, 'temporal_mixing': True, 'sequence_noise': True, 'stoch_scale': 1.5}),
        ('full_random', {'sequence_noise': True}),
    ]
    
    batch_size = 32
    num_samples = 5
    
    results = {}
    
    for mode, kwargs in test_configs:
        config_name = f"{mode}"
        if kwargs:
            config_name += f"_{'+'.join(k for k, v in kwargs.items() if v)}"
            if 'stoch_scale' in kwargs:
                config_name += f"_scale{kwargs['stoch_scale']}"
        
        print(f"\n🔍 Testing: {config_name}")
        
        config = MockConfig(mode, **kwargs)
        trainer = MockTrainer(config)
        
        # Generate multiple samples to see diversity
        samples = []
        for i in range(num_samples):
            trainer.current_step = 1000 + i  # Simulate different steps
            t = trainer.sample_timesteps(batch_size, mode='training', batch_idx=i)
            samples.append(t.numpy())
        
        samples = np.array(samples)
        
        # Calculate statistics
        mean_val = np.mean(samples)
        std_val = np.std(samples)
        min_val = np.min(samples)
        max_val = np.max(samples)
        
        # Calculate diversity between samples
        sample_means = np.mean(samples, axis=1)
        diversity = np.std(sample_means)
        
        results[config_name] = {
            'samples': samples,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'diversity': diversity,
        }
        
        print(f"   📊 Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        print(f"   📊 Range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"   📊 Sample diversity: {diversity:.4f}")
        
        # Show first few values from first sample
        first_sample = samples[0][:5]
        print(f"   📊 First 5 values: {first_sample}")
    
    print("\n" + "=" * 60)
    print("📋 Summary of Stochasticity Modes:")
    print()
    
    for config_name, stats in results.items():
        print(f"🔹 {config_name}:")
        print(f"   Diversity: {stats['diversity']:.4f} (higher = more stochastic)")
        print(f"   Coverage: [{stats['min']:.3f}, {stats['max']:.3f}] (wider = better)")
        print()
    
    # Recommendations
    print("🎯 Recommendations:")
    print()
    print("📌 For **maximum reproducibility**: Use 'deterministic' mode")
    print("📌 For **distributed training**: Use 'rank_based' mode (current default)")
    print("📌 For **enhanced exploration**: Use 'enhanced_stochastic' with extra_entropy=True")
    print("📌 For **maximum stochasticity**: Use 'full_random' (breaks reproducibility)")
    print("📌 For **fine-tuning**: Adjust 'stochasticity_scale' (0.5-2.0 range)")
    
    print("\n🔧 Configuration Examples:")
    print()
    print("# Conservative enhancement (recommended)")
    print("time_sampling:")
    print("  mode: 'enhanced_stochastic'")
    print("  extra_entropy_sources: true")
    print("  temporal_mixing: false")
    print("  sequence_level_noise: false")
    print("  stochasticity_scale: 1.0")
    print()
    print("# Aggressive enhancement")
    print("time_sampling:")
    print("  mode: 'enhanced_stochastic'")
    print("  extra_entropy_sources: true")
    print("  temporal_mixing: true")
    print("  sequence_level_noise: true")
    print("  stochasticity_scale: 1.5")
    print()
    print("# Maximum chaos (for experimentation)")
    print("time_sampling:")
    print("  mode: 'full_random'")
    print("  sequence_level_noise: true")
    print("  stochasticity_scale: 2.0")

def main():
    """Run the stochasticity test."""
    try:
        test_stochasticity_modes()
        print("\n🎉 All tests completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
