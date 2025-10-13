"""
Data utilities for protein sequence processing and tokenization.
"""
import torch
import json
import numpy as np
from transformers import GPT2TokenizerFast
from torch.utils.data import Dataset
from pathlib import Path


class ProteinTokenizer:
    """Protein tokenizer based on the download script."""

    def __init__(self, vocab_file=None, merges_file=None):
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        self.special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        
        # Create vocabulary
        self.vocab = {}
        self.vocab.update({token: i for i, token in enumerate(self.special_tokens)})
        self.vocab.update({aa: i + len(self.special_tokens) for i, aa in enumerate(self.amino_acids)})
        
        # Create reverse mapping
        self.id_to_token = {i: token for token, i in self.vocab.items()}
        
        # Special token IDs
        self.pad_token_id = self.vocab["<pad>"]
        self.unk_token_id = self.vocab["<unk>"]
        self.mask_token_id = self.vocab["<mask>"]
        self.bos_token_id = self.vocab["<s>"]
        self.eos_token_id = self.vocab["</s>"]
        
        # Use GPT2 tokenizer if files provided
        if vocab_file and merges_file:
            try:
                self.tokenizer = GPT2TokenizerFast(
                    vocab_file=vocab_file,
                    merges_file=merges_file,
                    unk_token="<unk>",
                    bos_token="<s>",
                    eos_token="</s>",
                    pad_token="<pad>",
                    mask_token="<mask>"
                )
                self.use_gpt2_tokenizer = True
                print("✅ Using GPT2 tokenizer")
            except Exception as e:
                print(f"⚠️  Failed to load GPT2 tokenizer: {e}")
                print("🔄 Falling back to amino acid tokenizer")
                self.use_gpt2_tokenizer = False
        else:
            self.use_gpt2_tokenizer = False
            print("✅ Using amino acid tokenizer")

    def encode(self, sequence, max_length=512, add_special_tokens=True):
        """Encode a protein sequence to token IDs."""
        if self.use_gpt2_tokenizer:
            return self._encode_gpt2(sequence, max_length, add_special_tokens)
        else:
            return self._encode_amino_acid(sequence, max_length, add_special_tokens)

    def _encode_gpt2(self, sequence, max_length, add_special_tokens):
        """Encode using GPT2 tokenizer."""
        tokens = self.tokenizer.encode(
            sequence,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=add_special_tokens
        )
        return torch.tensor(tokens, dtype=torch.long)

    def _encode_amino_acid(self, sequence, max_length, add_special_tokens):
        """Encode using amino acid tokenizer."""
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        
        # Convert sequence to tokens
        for char in sequence:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.unk_token_id)
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        
        # Truncate if too long
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.eos_token_id] if add_special_tokens else tokens[:max_length]
        
        # Pad if too short
        if len(tokens) < max_length:
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))

        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, token_ids):
        """Decode token IDs back to sequence."""
        if self.use_gpt2_tokenizer:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            tokens = [self.id_to_token.get(int(id), "<unk>") for id in token_ids]
            # Remove special tokens and join
            sequence = ''.join([t for t in tokens if t not in self.special_tokens])
            return sequence

    @property
    def vocab_size(self):
        """Get vocabulary size."""
        if self.use_gpt2_tokenizer:
            return len(self.tokenizer)
        else:
            return len(self.vocab)


class UniRef50Dataset(Dataset):
    """Dataset class for processed UniRef50 data - supports both tokenized and untokenized data."""

    def __init__(self, data_file, tokenize_on_fly=False, max_length=512, use_streaming=False):
        self.data_file = data_file
        self.tokenize_on_fly = tokenize_on_fly
        self.max_length = max_length
        self.use_streaming = use_streaming
        
        if tokenize_on_fly:
            self.tokenizer = ProteinTokenizer()
        
        if not use_streaming:
            self._load_data()
        else:
            self._setup_streaming()

    def _load_data(self):
        """Load all data into memory."""
        print(f"📂 Loading data from {self.data_file}")
        
        if self.data_file.endswith('.json'):
            with open(self.data_file, 'r') as f:
                self.data = json.load(f)
        elif self.data_file.endswith('.jsonl'):
            self.data = []
            with open(self.data_file, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))
        elif self.data_file.endswith('.pt'):
            # PyTorch file - load with progress for large files
            import os
            file_size_mb = os.path.getsize(self.data_file) / (1024 * 1024)
            print(f"📊 File size: {file_size_mb:.1f} MB")

            if file_size_mb > 100:  # Large file
                print("⚠️  Large file detected, this may take a while...")

            self.data = torch.load(self.data_file, map_location='cpu')
        else:
            raise ValueError(f"Unsupported file format: {self.data_file}")

        print(f"✅ Loaded {len(self.data)} sequences")

    def _setup_streaming(self):
        """Setup for streaming data loading."""
        print(f"🌊 Setting up streaming from {self.data_file}")

        # Check if it's a .pt file (binary) or text file
        if self.data_file.endswith('.pt'):
            print("⚠️  .pt files don't support streaming, loading into memory instead")
            self._load_data()
            return

        # Count lines for length estimation (only for text files)
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.length = sum(1 for _ in f)
            print(f"✅ Streaming setup complete, estimated {self.length} sequences")
        except UnicodeDecodeError:
            print("⚠️  File encoding issue, falling back to memory loading")
            self._load_data()

    def __len__(self):
        if self.use_streaming:
            return self.length
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.use_streaming:
            return self._get_streaming_item(idx)
        else:
            return self._get_memory_item(idx)

    def _get_memory_item(self, idx):
        """Get item from memory-loaded data."""
        try:
            item = self.data[idx]
            
            if self.tokenize_on_fly:
                # Tokenize on the fly
                if isinstance(item, dict):
                    sequence = item.get('sequence', item.get('text', ''))
                else:
                    sequence = str(item)
                
                tokens = self.tokenizer.encode(sequence, max_length=self.max_length)
                return tokens
            else:
                # Data is already tokenized
                if isinstance(item, dict):
                    tokens = item.get('tokens', item.get('input_ids', []))
                else:
                    tokens = item
                
                # Ensure it's a tensor
                if not isinstance(tokens, torch.Tensor):
                    tokens = torch.tensor(tokens, dtype=torch.long)
                
                # Pad or truncate to max_length
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                elif len(tokens) < self.max_length:
                    pad_length = self.max_length - len(tokens)
                    tokens = torch.cat([tokens, torch.zeros(pad_length, dtype=torch.long)])
                
                return tokens
                
        except Exception as e:
            print(f"❌ Error loading memory item {idx}: {e}")
            # Return a dummy sequence
            return torch.zeros(self.max_length, dtype=torch.long)

    def _get_streaming_item(self, idx):
        """Get item from streaming data."""
        try:
            # For .pt files, we can't really stream, so this shouldn't be called
            if self.data_file.endswith('.pt'):
                raise ValueError("Streaming not supported for .pt files")

            with open(self.data_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == idx:
                        item = json.loads(line.strip())
                        break
                else:
                    raise IndexError(f"Index {idx} out of range")
            
            if self.tokenize_on_fly:
                # Tokenize on the fly
                if isinstance(item, dict):
                    sequence = item.get('sequence', item.get('text', ''))
                else:
                    sequence = str(item)
                
                tokens = self.tokenizer.encode(sequence, max_length=self.max_length)
                return tokens
            else:
                # Data is already tokenized
                if isinstance(item, dict):
                    tokens = item.get('tokens', item.get('input_ids', []))
                else:
                    tokens = item
                
                # Ensure it's a tensor
                if not isinstance(tokens, torch.Tensor):
                    tokens = torch.tensor(tokens, dtype=torch.long)
                
                # Pad or truncate to max_length
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                elif len(tokens) < self.max_length:
                    pad_length = self.max_length - len(tokens)
                    tokens = torch.cat([tokens, torch.zeros(pad_length, dtype=torch.long)])
                
                return tokens

        except Exception as e:
            print(f"❌ Error loading streaming item {idx}: {e}")
            # Return a dummy sequence
            return torch.zeros(self.max_length, dtype=torch.long)


def safe_getattr(obj, path, default=None):
    """Safely get nested attributes with default fallback."""
    try:
        parts = path.split('.')
        current = obj
        for part in parts:
            current = getattr(current, part)
        return current
    except AttributeError:
        return default
