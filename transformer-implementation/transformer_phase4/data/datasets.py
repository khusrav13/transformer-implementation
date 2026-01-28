"""
Datasets for Transformer Training

Implements:
1. CopyTask - Simple sequence copying (for testing)
2. ReversalTask - Reverse sequence (slightly harder)
3. SortingTask - Sort numbers
4. SimpleTranslationDataset - Basic translation pairs
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import random


class CopyTaskDataset(Dataset):
    """
    Copy Task Dataset

    The model learns to copy input sequence to output.
    This is the simplest task to verify the model works.

    Example:
        Input:  [BOS, 5, 3, 8, 2, EOS, PAD, PAD]
        Output: [BOS, 5, 3, 8, 2, EOS, PAD, PAD]

    Args:
        num_samples: Number of samples to generate
        seq_length: Length of sequences (excluding special tokens)
        vocab_size: Size of vocabulary (tokens 3 onwards are data tokens)
        pad_idx: Padding token index
        bos_idx: Beginning of sequence token index
        eos_idx: End of sequence token index
    """

    def __init__(
        self,
        num_samples: int = 10000,
        seq_length: int = 10,
        vocab_size: int = 100,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # Pre-generate all samples
        self.data = self._generate_data()

    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate all samples"""
        data = []
        for _ in range(self.num_samples):
            # Random sequence length (variable for better generalization)
            length = random.randint(self.seq_length // 2, self.seq_length)

            # Generate random tokens (avoid special tokens 0, 1, 2)
            tokens = torch.randint(3, self.vocab_size, (length,))

            # Create source: [BOS, tokens, EOS, PAD...]
            src = torch.full((self.seq_length + 2,), self.pad_idx, dtype=torch.long)
            src[0] = self.bos_idx
            src[1:length + 1] = tokens
            src[length + 1] = self.eos_idx

            # Target is same as source (copy task)
            tgt = src.clone()

            data.append((src, tgt))

        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src, tgt = self.data[idx]
        # Labels are shifted target (predict next token)
        labels = tgt[1:].clone()  # Remove BOS, keep rest as labels
        return {
            'src': src,
            'tgt': tgt,
            'labels': labels
        }


class ReversalTaskDataset(Dataset):
    """
    Reversal Task Dataset

    The model learns to reverse the input sequence.
    Slightly harder than copy task.

    Example:
        Input:  [BOS, 5, 3, 8, 2, EOS]
        Output: [BOS, 2, 8, 3, 5, EOS]
    """

    def __init__(
        self,
        num_samples: int = 10000,
        seq_length: int = 10,
        vocab_size: int = 100,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.data = self._generate_data()

    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        data = []
        for _ in range(self.num_samples):
            length = random.randint(self.seq_length // 2, self.seq_length)
            tokens = torch.randint(3, self.vocab_size, (length,))

            # Source
            src = torch.full((self.seq_length + 2,), self.pad_idx, dtype=torch.long)
            src[0] = self.bos_idx
            src[1:length + 1] = tokens
            src[length + 1] = self.eos_idx

            # Target: reversed tokens
            tgt = torch.full((self.seq_length + 2,), self.pad_idx, dtype=torch.long)
            tgt[0] = self.bos_idx
            tgt[1:length + 1] = tokens.flip(0)  # Reverse
            tgt[length + 1] = self.eos_idx

            data.append((src, tgt))
        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src, tgt = self.data[idx]
        labels = tgt[1:].clone()
        return {'src': src, 'tgt': tgt, 'labels': labels}


class SortingTaskDataset(Dataset):
    """
    Sorting Task Dataset

    The model learns to sort the input sequence.
    Tests whether transformer can learn algorithmic patterns.

    Example:
        Input:  [BOS, 5, 3, 8, 2, EOS]
        Output: [BOS, 2, 3, 5, 8, EOS]
    """

    def __init__(
        self,
        num_samples: int = 10000,
        seq_length: int = 8,
        max_value: int = 50,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.max_value = max_value
        self.vocab_size = max_value + 3  # +3 for special tokens
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.data = self._generate_data()

    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        data = []
        for _ in range(self.num_samples):
            length = random.randint(3, self.seq_length)
            # Generate random numbers (offset by 3 to avoid special tokens)
            tokens = torch.randint(3, self.max_value + 3, (length,))

            # Source
            src = torch.full((self.seq_length + 2,), self.pad_idx, dtype=torch.long)
            src[0] = self.bos_idx
            src[1:length + 1] = tokens
            src[length + 1] = self.eos_idx

            # Target: sorted tokens
            sorted_tokens, _ = tokens.sort()
            tgt = torch.full((self.seq_length + 2,), self.pad_idx, dtype=torch.long)
            tgt[0] = self.bos_idx
            tgt[1:length + 1] = sorted_tokens
            tgt[length + 1] = self.eos_idx

            data.append((src, tgt))
        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src, tgt = self.data[idx]
        labels = tgt[1:].clone()
        return {'src': src, 'tgt': tgt, 'labels': labels}


class SimpleTranslationDataset(Dataset):
    """
    Simple Translation Dataset

    Uses predefined word pairs for basic translation testing.
    Good for verifying the model can learn simple mappings.

    Example (English to "Pig Latin"):
        Input:  "hello world"
        Output: "ellohay orldway"
    """

    def __init__(
        self,
        num_samples: int = 10000,
        max_seq_length: int = 20,
        vocab_size: int = 1000,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2
    ):
        self.num_samples = num_samples
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # Create simple mapping (token i+offset maps to token i+offset+vocab_size//2)
        self.offset = 3
        self.mapping_offset = (vocab_size - 3) // 2

        self.data = self._generate_data()

    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        data = []
        for _ in range(self.num_samples):
            length = random.randint(3, self.max_seq_length - 2)

            # Source tokens
            src_tokens = torch.randint(
                self.offset,
                self.offset + self.mapping_offset,
                (length,)
            )

            # Target tokens (simple offset mapping)
            tgt_tokens = src_tokens + self.mapping_offset

            # Create padded sequences
            src = torch.full((self.max_seq_length,), self.pad_idx, dtype=torch.long)
            src[0] = self.bos_idx
            src[1:length + 1] = src_tokens
            src[length + 1] = self.eos_idx

            tgt = torch.full((self.max_seq_length,), self.pad_idx, dtype=torch.long)
            tgt[0] = self.bos_idx
            tgt[1:length + 1] = tgt_tokens
            tgt[length + 1] = self.eos_idx

            data.append((src, tgt))
        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src, tgt = self.data[idx]
        labels = tgt[1:].clone()
        return {'src': src, 'tgt': tgt, 'labels': labels}


class ArithmeticDataset(Dataset):
    """
    Simple Arithmetic Dataset

    The model learns to perform simple addition.

    Example:
        Input:  "12+34="  -> [BOS, 1, 2, +, 3, 4, =, EOS]
        Output: "46"      -> [BOS, 4, 6, EOS]

    Uses vocab: 0=PAD, 1=BOS, 2=EOS, 3-12=digits, 13=+, 14==
    """

    def __init__(
        self,
        num_samples: int = 10000,
        max_digits: int = 2,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2
    ):
        self.num_samples = num_samples
        self.max_digits = max_digits
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # Vocab: 0=PAD, 1=BOS, 2=EOS, 3-12=digits(0-9), 13=+, 14==
        self.digit_offset = 3
        self.plus_token = 13
        self.equals_token = 14
        self.vocab_size = 15

        self.max_len = 2 * max_digits + 4  # digits + plus + equals + BOS + EOS
        self.data = self._generate_data()

    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        data = []
        max_num = 10 ** self.max_digits - 1

        for _ in range(self.num_samples):
            a = random.randint(0, max_num)
            b = random.randint(0, max_num)
            result = a + b

            # Convert to token sequences
            src_tokens = self._num_to_tokens(a) + [self.plus_token] + \
                        self._num_to_tokens(b) + [self.equals_token]
            tgt_tokens = self._num_to_tokens(result)

            # Create padded tensors
            src = torch.full((self.max_len,), self.pad_idx, dtype=torch.long)
            tgt = torch.full((self.max_len,), self.pad_idx, dtype=torch.long)

            src[0] = self.bos_idx
            for i, t in enumerate(src_tokens):
                src[i + 1] = t
            src[len(src_tokens) + 1] = self.eos_idx

            tgt[0] = self.bos_idx
            for i, t in enumerate(tgt_tokens):
                tgt[i + 1] = t
            tgt[len(tgt_tokens) + 1] = self.eos_idx

            data.append((src, tgt))
        return data

    def _num_to_tokens(self, num: int) -> List[int]:
        """Convert number to list of digit tokens"""
        if num == 0:
            return [self.digit_offset]
        tokens = []
        while num > 0:
            tokens.append(num % 10 + self.digit_offset)
            num //= 10
        return tokens[::-1]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src, tgt = self.data[idx]
        labels = tgt[1:].clone()
        return {'src': src, 'tgt': tgt, 'labels': labels}


def create_dataloader(
    dataset_type: str = 'copy',
    num_samples: int = 10000,
    batch_size: int = 32,
    seq_length: int = 10,
    vocab_size: int = 100,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> Tuple[DataLoader, Dataset]:
    """
    Factory function to create data loader

    Args:
        dataset_type: 'copy', 'reversal', 'sorting', 'translation', 'arithmetic'
        num_samples: Number of samples
        batch_size: Batch size
        seq_length: Sequence length
        vocab_size: Vocabulary size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        **kwargs: Additional dataset arguments

    Returns:
        DataLoader and Dataset
    """
    if dataset_type == 'copy':
        dataset = CopyTaskDataset(
            num_samples=num_samples,
            seq_length=seq_length,
            vocab_size=vocab_size,
            **kwargs
        )
    elif dataset_type == 'reversal':
        dataset = ReversalTaskDataset(
            num_samples=num_samples,
            seq_length=seq_length,
            vocab_size=vocab_size,
            **kwargs
        )
    elif dataset_type == 'sorting':
        dataset = SortingTaskDataset(
            num_samples=num_samples,
            seq_length=seq_length,
            **kwargs
        )
    elif dataset_type == 'translation':
        dataset = SimpleTranslationDataset(
            num_samples=num_samples,
            max_seq_length=seq_length + 2,
            vocab_size=vocab_size,
            **kwargs
        )
    elif dataset_type == 'arithmetic':
        dataset = ArithmeticDataset(
            num_samples=num_samples,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader, dataset
