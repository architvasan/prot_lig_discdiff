"""
ESM-based protein sequence evaluation utilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union
import warnings

try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    warnings.warn("ESM not available. Install with: pip install fair-esm")


class ESMEvaluator:
    """ESM-based protein sequence evaluator for perplexity calculation."""
    
    def __init__(self, model_name: str = "esm2_t6_8M_UR50D", device: Optional[str] = None):
        """
        Initialize ESM evaluator.
        
        Args:
            model_name: ESM model name (esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, etc.)
            device: Device to run on (auto-detect if None)
        """
        if not ESM_AVAILABLE:
            raise ImportError("ESM not available. Install with: pip install fair-esm")
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Load ESM model
        print(f"🔬 Loading ESM model: {model_name}")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get batch converter
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Special tokens
        self.mask_idx = self.alphabet.mask_idx
        self.cls_idx = self.alphabet.cls_idx
        self.eos_idx = self.alphabet.eos_idx
        self.pad_idx = self.alphabet.padding_idx
        
        print(f"✅ ESM model loaded on {self.device}")
    
    def calculate_perplexity(self, 
                           sequences: List[str], 
                           batch_size: int = 4,
                           mask_fraction: float = 0.15,
                           seed: int = 42) -> List[float]:
        """
        Calculate ESM perplexity for protein sequences using masked language modeling.
        
        Args:
            sequences: List of protein sequences
            batch_size: Batch size for processing
            mask_fraction: Fraction of tokens to mask for MLM
            seed: Random seed for reproducible masking
            
        Returns:
            List of perplexity values for each sequence
        """
        if not sequences:
            return []
        
        # Clean sequences (remove special tokens)
        clean_sequences = []
        for seq in sequences:
            clean_seq = seq.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
            # Remove any non-amino acid characters
            clean_seq = ''.join(c for c in clean_seq if c in 'ACDEFGHIKLMNPQRSTVWY')
            if clean_seq:
                clean_sequences.append(clean_seq)
        
        if not clean_sequences:
            print("⚠️  No valid sequences found for ESM evaluation")
            return []
        
        print(f"📊 Calculating ESM perplexity for {len(clean_sequences)} sequences...")
        
        perplexities = []
        
        # Process in batches
        for i in range(0, len(clean_sequences), batch_size):
            batch_seqs = clean_sequences[i:i + batch_size]
            batch_perplexities = self._calculate_batch_perplexity(
                batch_seqs, mask_fraction, seed + i
            )
            perplexities.extend(batch_perplexities)
        
        return perplexities
    
    def _calculate_batch_perplexity(self, 
                                  sequences: List[str], 
                                  mask_fraction: float,
                                  seed: int) -> List[float]:
        """Calculate perplexity for a batch of sequences."""
        if not sequences:
            return []
        
        # Prepare data for ESM
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        perplexities = []
        
        with torch.no_grad():
            for seq_idx, (label, seq_str) in enumerate(zip(batch_labels, batch_strs)):
                try:
                    # Get tokens for this sequence
                    seq_tokens = batch_tokens[seq_idx:seq_idx+1]  # [1, seq_len]
                    seq_len = len(seq_str)
                    
                    if seq_len == 0:
                        perplexities.append(float('inf'))
                        continue
                    
                    # Create multiple masked versions for robust perplexity estimation
                    total_log_likelihood = 0.0
                    total_tokens = 0
                    num_masks = max(1, int(seq_len * mask_fraction))
                    
                    # Generate multiple random masks
                    torch.manual_seed(seed + seq_idx)
                    for mask_iter in range(5):  # Average over 5 different masks
                        # Create masked version
                        masked_tokens = seq_tokens.clone()
                        
                        # Randomly select positions to mask (excluding CLS and EOS)
                        maskable_positions = list(range(1, seq_len + 1))  # Skip CLS token
                        if len(maskable_positions) == 0:
                            continue
                        
                        mask_positions = torch.randperm(len(maskable_positions))[:num_masks]
                        mask_positions = [maskable_positions[i] for i in mask_positions]
                        
                        # Store original tokens for loss calculation
                        original_tokens = seq_tokens[0, mask_positions].clone()
                        
                        # Apply masks
                        for pos in mask_positions:
                            masked_tokens[0, pos] = self.mask_idx
                        
                        # Forward pass
                        outputs = self.model(masked_tokens)
                        logits = outputs['logits']  # [1, seq_len, vocab_size]
                        
                        # Calculate log probabilities for masked positions
                        log_probs = F.log_softmax(logits[0], dim=-1)  # [seq_len, vocab_size]
                        
                        # Extract log probabilities for original tokens at masked positions
                        for i, pos in enumerate(mask_positions):
                            original_token = original_tokens[i].item()
                            log_prob = log_probs[pos, original_token].item()
                            total_log_likelihood += log_prob
                            total_tokens += 1
                    
                    # Calculate perplexity
                    if total_tokens > 0:
                        avg_log_likelihood = total_log_likelihood / total_tokens
                        perplexity = torch.exp(-torch.tensor(avg_log_likelihood)).item()
                    else:
                        perplexity = float('inf')
                    
                    perplexities.append(perplexity)
                    
                except Exception as e:
                    print(f"⚠️  Error calculating perplexity for sequence {seq_idx}: {e}")
                    perplexities.append(float('inf'))
        
        return perplexities
    
    def calculate_sequence_likelihood(self, sequence: str) -> float:
        """
        Calculate the likelihood of a sequence under the ESM model.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Log likelihood of the sequence
        """
        # Clean sequence
        clean_seq = sequence.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
        clean_seq = ''.join(c for c in clean_seq if c in 'ACDEFGHIKLMNPQRSTVWY')
        
        if not clean_seq:
            return float('-inf')
        
        # Prepare data
        data = [("seq", clean_seq)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(batch_tokens)
            logits = outputs['logits'][0]  # [seq_len, vocab_size]
            
            # Calculate log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Sum log probabilities for the sequence (excluding special tokens)
            total_log_prob = 0.0
            for i in range(1, len(clean_seq) + 1):  # Skip CLS token
                token_id = batch_tokens[0, i].item()
                if token_id not in [self.cls_idx, self.eos_idx, self.pad_idx]:
                    total_log_prob += log_probs[i, token_id].item()
            
            return total_log_prob


def calculate_esm_perplexity(sequences: List[str], 
                           model_name: str = "esm2_t6_8M_UR50D",
                           batch_size: int = 4,
                           mask_fraction: float = 0.15,
                           seed: int = 42,
                           device: Optional[str] = None) -> List[float]:
    """
    Convenience function to calculate ESM perplexity for sequences.
    
    Args:
        sequences: List of protein sequences
        model_name: ESM model name
        batch_size: Batch size for processing
        mask_fraction: Fraction of tokens to mask
        seed: Random seed
        device: Device to run on
        
    Returns:
        List of perplexity values
    """
    if not ESM_AVAILABLE:
        print("⚠️  ESM not available, returning dummy perplexities")
        return [float('inf')] * len(sequences)
    
    try:
        evaluator = ESMEvaluator(model_name=model_name, device=device)
        return evaluator.calculate_perplexity(
            sequences=sequences,
            batch_size=batch_size,
            mask_fraction=mask_fraction,
            seed=seed
        )
    except Exception as e:
        print(f"⚠️  Error in ESM perplexity calculation: {e}")
        return [float('inf')] * len(sequences)
