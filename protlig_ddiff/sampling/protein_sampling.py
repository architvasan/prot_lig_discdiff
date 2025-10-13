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
            
            # Apply conditioning
            for i, loc in enumerate(input_locs):
                if loc < x.shape[1]:  # Make sure we don't go out of bounds
                    x[:, loc] = ids_tensor[:, i]
            
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
            samples = sampling_fn(self.model)
        
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
            prefix_tokens = [self.tokenizer.token_to_id.get(char, 0) for char in prefix]
        
        if suffix:
            suffix_tokens = [self.tokenizer.token_to_id.get(char, 0) for char in suffix]
        
        # Add BOS/EOS if not already present
        if not prefix_tokens or prefix_tokens[0] != self.tokenizer.token_to_id.get('<s>', 1):
            prefix_tokens = [self.tokenizer.token_to_id.get('<s>', 1)] + prefix_tokens
        
        if not suffix_tokens or suffix_tokens[-1] != self.tokenizer.token_to_id.get('</s>', 2):
            suffix_tokens = suffix_tokens + [self.tokenizer.token_to_id.get('</s>', 2)]
        
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
    
    # Get sampling config
    sampling_config = getattr(config, 'sampling', None)
    if sampling_config is None:
        # Default sampling config
        batch_size = 4
        max_length = 128
        steps = 256
        predictor = 'analytic'
    else:
        batch_size = getattr(sampling_config, 'eval_batch_size', 4)
        max_length = getattr(sampling_config, 'eval_max_length', 128)
        steps = getattr(sampling_config, 'eval_steps', 256)
        predictor = getattr(sampling_config, 'predictor', 'analytic')
    
    # Create sampler
    sampler = ProteinSampler(model, graph, noise, device=device)
    
    # Sample sequences
    try:
        sequences = sampler.sample_unconditional(
            batch_size=batch_size,
            max_length=max_length,
            steps=steps,
            predictor=predictor
        )
        
        print(f"\n🧬 Sampled sequences at step {step}:")
        for i, seq in enumerate(sequences):
            # Clean up sequence for display
            clean_seq = seq.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
            print(f"  Sample {i+1}: {clean_seq[:50]}{'...' if len(clean_seq) > 50 else ''}")
        
        return sequences
        
    except Exception as e:
        print(f"⚠️  Sampling failed at step {step}: {e}")
        return []
