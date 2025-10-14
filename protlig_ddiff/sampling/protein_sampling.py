"""
Protein-specific sampling with conditioning for start/end tokens.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Callable
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from protlig_ddiff.sampling.sampling import get_pc_sampler
from protlig_ddiff.utils.data_utils import ProteinTokenizer


class ProteinSampler:
    """Protein sequence sampler with conditioning support."""
    
    def __init__(self, model, graph, noise, tokenizer=None, device='cuda'):
        self.model = model
        self.graph = graph
        self.noise = noise
        self.device = device
        
        # Initialize tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = ProteinTokenizer()
        else:
            self.tokenizer = tokenizer
    
    def create_conditioning_function(self, 
                                   start_tokens: Optional[List[int]] = None,
                                   end_tokens: Optional[List[int]] = None,
                                   max_length: int = 512) -> Callable:
        """
        Create a conditioning function for start/end tokens.
        
        Args:
            start_tokens: List of token IDs to condition at the beginning
            end_tokens: List of token IDs to condition at the end
            max_length: Maximum sequence length
            
        Returns:
            Conditioning function that can be used with the sampler
        """
        
        # Default to BOS and EOS tokens if not specified
        if start_tokens is None:
            start_tokens = [self.tokenizer.token_to_id.get('<s>', 1)]  # BOS token
        if end_tokens is None:
            end_tokens = [self.tokenizer.token_to_id.get('</s>', 2)]   # EOS token
        
        # Create input_ids and input_locs for conditioning
        input_ids = start_tokens + end_tokens
        input_locs = (list(range(len(start_tokens))) + 
                     list(range(max_length - len(end_tokens), max_length)))
        
        def conditioning_fn(x):
            """Apply conditioning to the sequence."""
            batch_size = x.shape[0]
            
            # Convert to tensor if needed
            if not isinstance(input_ids, torch.Tensor):
                ids_tensor = torch.tensor(input_ids, device=self.device)
            else:
                ids_tensor = input_ids.to(self.device)
            
            # Repeat for batch
            ids_tensor = ids_tensor[None].repeat(batch_size, 1)
            x[:, input_locs] = ids_tensor
            ## Apply conditioning
            #for i, loc in enumerate(input_locs):
            #    if loc < x.shape[1]:  # Make sure we don't go out of bounds
            #        x[:, loc] = ids_tensor[:, i]
            
            return x
        
        return conditioning_fn
    
    def sample_sequences(self,
                        batch_size: int = 1,
                        max_length: int = 512,
                        steps: int = 1024,
                        start_tokens: Optional[List[int]] = None,
                        end_tokens: Optional[List[int]] = None,
                        predictor: str = 'analytic',
                        denoise: bool = True,
                        eps: float = 1e-5) -> torch.Tensor:
        """
        Sample protein sequences with conditioning.
        
        Args:
            batch_size: Number of sequences to sample
            max_length: Maximum sequence length
            steps: Number of sampling steps
            start_tokens: Tokens to condition at the beginning
            end_tokens: Tokens to condition at the end
            predictor: Sampling predictor ('analytic', 'euler', etc.)
            denoise: Whether to apply denoising step
            eps: Small epsilon for numerical stability
            
        Returns:
            Tensor of sampled sequences [batch_size, max_length]
        """
        
        # Create conditioning function
        conditioning_fn = self.create_conditioning_function(
            start_tokens=start_tokens,
            end_tokens=end_tokens,
            max_length=max_length
        )
        
        # Get the sampling function
        sampling_fn = get_pc_sampler(
            graph=self.graph,
            noise=self.noise,
            batch_dims=(batch_size, max_length),
            predictor=predictor,
            steps=steps,
            denoise=denoise,
            eps=eps,
            device=self.device,
            proj_fun=conditioning_fn
        )
        
        # Sample sequences
        with torch.no_grad():
            samples = conditioning_fn(sampling_fn(self.model))
        
        return samples
    
    def sample_with_prefix_suffix(self,
                                 prefix: str = "",
                                 suffix: str = "",
                                 batch_size: int = 1,
                                 max_length: int = 512,
                                 steps: int = 1024,
                                 predictor: str = 'analytic') -> List[str]:
        """
        Sample sequences with text prefix and suffix.
        
        Args:
            prefix: Text prefix (will be tokenized)
            suffix: Text suffix (will be tokenized)
            batch_size: Number of sequences to sample
            max_length: Maximum sequence length
            steps: Number of sampling steps
            predictor: Sampling predictor
            
        Returns:
            List of decoded protein sequences
        """
        
        # Tokenize prefix and suffix
        prefix_tokens = []
        suffix_tokens = []
        
        if prefix:
            # For protein sequences, we might want to tokenize character by character
            # or use the tokenizer's encode method
            prefix_tokens = [self.tokenizer.vocab.get(char, 0) for char in prefix]

        if suffix:
            suffix_tokens = [self.tokenizer.vocab.get(char, 0) for char in suffix]

        # Add BOS/EOS if not already present
        if not prefix_tokens or prefix_tokens[0] != self.tokenizer.vocab.get('<s>', 1):
            prefix_tokens = [self.tokenizer.vocab.get('<s>', 1)] + prefix_tokens

        if not suffix_tokens or suffix_tokens[-1] != self.tokenizer.vocab.get('</s>', 2):
            suffix_tokens = suffix_tokens + [self.tokenizer.vocab.get('</s>', 2)]
        
        # Sample sequences
        samples = self.sample_sequences(
            batch_size=batch_size,
            max_length=max_length,
            steps=steps,
            start_tokens=prefix_tokens,
            end_tokens=suffix_tokens,
            predictor=predictor
        )
        
        # Decode sequences
        decoded_sequences = []
        for i in range(batch_size):
            sequence = self.tokenizer.decode(samples[i].cpu().tolist())
            decoded_sequences.append(sequence)
        
        return decoded_sequences
    
    def sample_unconditional(self,
                           batch_size: int = 1,
                           max_length: int = 512,
                           steps: int = 1024,
                           predictor: str = 'analytic') -> List[str]:
        """
        Sample unconditional protein sequences (only BOS/EOS conditioning).
        
        Args:
            batch_size: Number of sequences to sample
            max_length: Maximum sequence length
            steps: Number of sampling steps
            predictor: Sampling predictor
            
        Returns:
            List of decoded protein sequences
        """
        
        return self.sample_with_prefix_suffix(
            prefix="",
            suffix="",
            batch_size=batch_size,
            max_length=max_length,
            steps=steps,
            predictor=predictor
        )


def create_protein_sampler(model, graph, noise, tokenizer=None, device='cuda'):
    """
    Factory function to create a ProteinSampler.
    
    Args:
        model: Trained diffusion model
        graph: Graph object (e.g., Absorbing)
        noise: Noise schedule object
        tokenizer: Optional tokenizer (will create default if None)
        device: Device to run sampling on
        
    Returns:
        ProteinSampler instance
    """
    return ProteinSampler(model, graph, noise, tokenizer, device)


def sample_during_training(model, graph, noise, config, step, device='cuda'):
    """
    Sample sequences during training for monitoring/logging.

    Args:
        model: Current model state
        graph: Graph object
        noise: Noise schedule
        config: Training config with sampling parameters
        step: Current training step
        device: Device to run on

    Returns:
        List of sampled protein sequences
    """

    # print(f"🔍 Debug: sample_during_training called at step {step}")

    # Get sampling config
    sampling_config = getattr(config, 'sampling', None)
    # print(f"🔍 Debug: sampling_config from config: {sampling_config}")
    # print(f"🔍 Debug: config type: {type(config)}")
    # print(f"🔍 Debug: config attributes: {dir(config) if hasattr(config, '__dict__') else 'no __dict__'}")

    if sampling_config is None:
        # print("🔍 Debug: Using default sampling config")
        # Default sampling config
        batch_size = 4
        max_length = 128
        steps = 256
        predictor = 'analytic'
    else:
        # Try different ways to access the config values
        if hasattr(sampling_config, 'eval_batch_size'):
            batch_size = sampling_config.eval_batch_size
        elif hasattr(sampling_config, 'get'):
            batch_size = sampling_config.get('eval_batch_size', 4)
        else:
            batch_size = getattr(sampling_config, 'eval_batch_size', 4)

        if hasattr(sampling_config, 'eval_max_length'):
            max_length = sampling_config.eval_max_length
        elif hasattr(sampling_config, 'get'):
            max_length = sampling_config.get('eval_max_length', 128)
        else:
            max_length = getattr(sampling_config, 'eval_max_length', 128)

        if hasattr(sampling_config, 'eval_steps'):
            steps = sampling_config.eval_steps
        elif hasattr(sampling_config, 'get'):
            steps = sampling_config.get('eval_steps', 256)
        else:
            steps = getattr(sampling_config, 'eval_steps', 256)

        if hasattr(sampling_config, 'predictor'):
            predictor = sampling_config.predictor
        elif hasattr(sampling_config, 'get'):
            predictor = sampling_config.get('predictor', 'analytic')
        else:
            predictor = getattr(sampling_config, 'predictor', 'analytic')

        # print(f"🔍 Debug: Using config - batch_size: {batch_size}, max_length: {max_length}, steps: {steps}, predictor: {predictor}")

    # Create sampler
    # print(f"🔍 Debug: Creating ProteinSampler...")
    sampler = ProteinSampler(model, graph, noise, device=device)

    # Sample sequences
    try:
        # print(f"🔍 Debug: Starting sampling...")
        sequences = sampler.sample_unconditional(
            batch_size=batch_size,
            max_length=max_length,
            steps=steps,
            predictor=predictor
        )

        # print(f"🔍 Debug: Sampling completed, got {len(sequences) if sequences else 0} sequences")

        # Validate sequences
        if sequences is None:
            print(f"❌ CRITICAL: Sampling returned None at step {step}")
            return None  # Return None to indicate critical failure

        if len(sequences) == 0:
            print(f"⚠️  Warning: Sampling returned empty list at step {step}")
            return []  # Return empty list to indicate no sequences generated

        # Check sequence quality
        valid_sequences = []
        for i, seq in enumerate(sequences):
            if seq is None or len(seq.strip()) == 0:
                print(f"⚠️  Warning: Empty or None sequence {i+1} at step {step}")
                continue
            valid_sequences.append(seq)

        if len(valid_sequences) == 0:
            print(f"⚠️  Warning: All sequences were invalid at step {step}")
            return []

        if len(valid_sequences) < len(sequences):
            print(f"⚠️  Warning: {len(sequences) - len(valid_sequences)} invalid sequences filtered out at step {step}")

        if valid_sequences:
            print(f"\n🧬 Sampled {len(valid_sequences)} valid sequences at step {step}:")
            for i, seq in enumerate(valid_sequences[:3]):  # Show first 3
                # Clean up sequence for display
                clean_seq = seq.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
                print(f"  Sample {i+1}: {clean_seq[:50]}{'...' if len(clean_seq) > 50 else ''}")

        return valid_sequences

    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ CUDA OOM during sampling at step {step}: {e}")
        print("🔧 Try reducing batch_size or max_length in sampling config")
        return None  # Critical failure

    except RuntimeError as e:
        if "CUDA" in str(e) or "device" in str(e).lower():
            print(f"❌ CUDA/Device error during sampling at step {step}: {e}")
            return None  # Critical failure
        elif "probability tensor contains" in str(e):
            print(f"⚠️  Numerical stability issue during sampling at step {step}: {e}")
            print(f"🔧 This is common in early training - the model will improve with more steps")

            # Try with reduced parameters for better stability
            try:
                print(f"🔧 Attempting sampling with reduced parameters...")
                reduced_sampler = ProteinSampler(model, graph, noise, device=device)
                reduced_sequences = reduced_sampler.sample_unconditional(
                    batch_size=min(batch_size, 2),  # Reduce batch size
                    max_length=min(max_length, 64),  # Reduce sequence length
                    steps=max(10, steps // 4),       # Reduce sampling steps
                    predictor=predictor
                )
                if reduced_sequences and len(reduced_sequences) > 0:
                    print(f"✅ Reduced parameter sampling succeeded with {len(reduced_sequences)} sequences")
                    return reduced_sequences
                else:
                    print(f"⚠️  Reduced parameter sampling also failed")
                    return []
            except Exception as e2:
                print(f"⚠️  Reduced parameter sampling failed: {e2}")
                return []  # Non-critical failure
        else:
            print(f"⚠️  Runtime error during sampling at step {step}: {e}")
            import traceback
            traceback.print_exc()
            return []  # Non-critical failure

    except Exception as e:
        print(f"⚠️  Unexpected error during sampling at step {step}: {e}")
        import traceback
        traceback.print_exc()
        return []  # Non-critical failure
