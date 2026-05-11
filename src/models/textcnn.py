"""
TextCNN for AG News text classification.
Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification.

Input : LongTensor of shape (B, seq_len)  -- token indices
Output: Tensor of shape (B, num_classes)  -- raw logits

Design mirrors the lightweight CNN_CIFAR used for CIFAR-100:
  - Fast enough for edge-device simulation (sequential CPU/GPU loop)
  - Small parameter count (~1M)
  - Compatible with the existing _local_train / _compute_logits pipeline
    without any modification (same forward(x) → logits interface)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.agnews import VOCAB_SIZE, PAD_IDX


class TextCNN(nn.Module):
    """
    Multi-kernel TextCNN.

    Args:
        vocab_size    : vocabulary size (including PAD=0, UNK=1)
        embed_dim     : embedding dimension
        num_classes   : number of output classes (4 for AG News)
        kernel_sizes  : list of convolutional kernel heights (n-gram sizes)
        num_filters   : number of filters per kernel size
        dropout       : dropout rate before the final classifier
        pad_idx       : padding index (embedding rows zeroed out)
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE + 2,   # +2 for PAD and UNK
        embed_dim: int = 128,
        num_classes: int = 4,
        kernel_sizes: tuple = (2, 3, 4),
        num_filters: int = 128,
        dropout: float = 0.5,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
            )
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor of shape (B, seq_len)
        Returns:
            Tensor of shape (B, num_classes)
        """
        # (B, seq_len) → (B, seq_len, embed_dim) → (B, embed_dim, seq_len)
        emb = self.embedding(x).permute(0, 2, 1)

        # Conv + ReLU + global max-pool for each kernel size
        pooled = []
        for conv in self.convs:
            # (B, num_filters, seq_len - k + 1)
            c = F.relu(conv(emb))
            # (B, num_filters)
            p = c.max(dim=2).values
            pooled.append(p)

        # (B, num_filters * n_kernels)
        cat = torch.cat(pooled, dim=1)
        out = self.fc(self.dropout(cat))
        return out
