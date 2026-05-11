"""
AG News dataset loading for KaaS-Edge (text modality).

Mirrors load_cifar100_safe_split() exactly:
  private_set, public_set, test_set = load_agnews_safe_split(...)

Each dataset item returns (input_ids, label):
  input_ids : torch.LongTensor of shape (SEQ_LEN,)  -- tokenized & padded
  label     : int in {0,1,2,3}  (World / Sports / Business / Sci/Tech)

The tokenizer is a simple whitespace tokenizer with a fixed vocabulary built
from the training split -- no external dependencies beyond torchtext / datasets.
Falls back to torchtext if available, otherwise uses HuggingFace datasets.
"""

import os
import re
import numpy as np
from collections import Counter
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, Subset

# ────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────

SEQ_LEN   = 64      # truncate / pad all sequences to this length
VOCAB_SIZE = 20000  # keep top-20k tokens (+ PAD=0, UNK=1)
PAD_IDX   = 0
UNK_IDX   = 1

# ────────────────────────────────────────────────────────────────
# Simple tokenizer (no external NLP library required)
# ────────────────────────────────────────────────────────────────

def _tokenize(text: str):
    """Lowercase + split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9']+", text.lower())


def build_vocab(texts, max_size=VOCAB_SIZE):
    """Build {token: idx} mapping from a list of raw strings."""
    counter = Counter()
    for t in texts:
        counter.update(_tokenize(t))
    # reserve 0=PAD, 1=UNK
    vocab = {tok: idx + 2 for idx, (tok, _) in
             enumerate(counter.most_common(max_size - 2))}
    return vocab


def encode(text: str, vocab: dict, seq_len: int = SEQ_LEN) -> torch.LongTensor:
    tokens = _tokenize(text)[:seq_len]
    ids = [vocab.get(t, UNK_IDX) for t in tokens]
    # pad to seq_len
    ids += [PAD_IDX] * (seq_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# ────────────────────────────────────────────────────────────────
# PyTorch Dataset wrapper
# ────────────────────────────────────────────────────────────────

class AGNewsDataset(Dataset):
    """
    Holds pre-encoded AG News samples as LongTensors.

    Args:
        texts  : list of raw strings (title + description)
        labels : list of ints (0-based)
        vocab  : token→index mapping (built from training split)
        seq_len: fixed sequence length after padding/truncation
    """

    def __init__(self, texts, labels, vocab, seq_len=SEQ_LEN):
        self.vocab   = vocab
        self.seq_len = seq_len
        self.encoded = [encode(t, vocab, seq_len) for t in texts]
        self.labels  = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encoded[idx], self.labels[idx]


# ────────────────────────────────────────────────────────────────
# Data loading helpers
# ────────────────────────────────────────────────────────────────

def _load_raw_agnews(root='./data'):
    """
    Load raw AG News texts & labels.
    Tries torchtext first, falls back to HuggingFace datasets.
    Labels are remapped to 0-based (torchtext uses 1-based).
    Returns (train_texts, train_labels, test_texts, test_labels).
    """
    cache_dir = os.path.join(root, 'agnews')
    os.makedirs(cache_dir, exist_ok=True)

    # ── Try torchtext ──────────────────────────────────────────
    try:
        from torchtext.datasets import AG_NEWS
        train_iter = AG_NEWS(root=cache_dir, split='train')
        test_iter  = AG_NEWS(root=cache_dir, split='test')
        train_labels, train_texts = zip(*[(lbl - 1, txt) for lbl, txt in train_iter])
        test_labels,  test_texts  = zip(*[(lbl - 1, txt) for lbl, txt in test_iter])
        print("[AG News] Loaded via torchtext.")
        return list(train_texts), list(train_labels), list(test_texts), list(test_labels)
    except Exception:
        pass

    # ── Fallback: HuggingFace datasets ────────────────────────
    try:
        from datasets import load_dataset
        ds = load_dataset("ag_news", cache_dir=cache_dir)

        def _extract(split):
            texts  = [row['text'] for row in ds[split]]
            labels = [row['label'] for row in ds[split]]
            return texts, labels

        train_texts, train_labels = _extract('train')
        test_texts,  test_labels  = _extract('test')
        print("[AG News] Loaded via HuggingFace datasets.")
        return train_texts, train_labels, test_texts, test_labels
    except Exception as e:
        raise RuntimeError(
            "Could not load AG News. Install torchtext or HuggingFace datasets:\n"
            "  pip install torchtext   OR   pip install datasets\n"
            f"Original error: {e}"
        )


# ────────────────────────────────────────────────────────────────
# Public API — mirrors load_cifar100_safe_split()
# ────────────────────────────────────────────────────────────────

def load_agnews_safe_split(
    root: str = './data',
    n_public: int = 5000,
    seed: int = 42,
    seq_len: int = SEQ_LEN,
    vocab_size: int = VOCAB_SIZE,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Safe split of AG News, mirroring load_cifar100_safe_split().

    Splits the training set (120k) into:
      - public_set  : n_public samples (server knowledge distillation)
      - private_set : remaining samples (client local training)
    Keeps the test set (7.6k) untouched.

    Returns:
        (private_set, public_set, test_set)
        Each item: (input_ids: LongTensor[seq_len], label: int)
    """
    train_texts, train_labels, test_texts, test_labels = _load_raw_agnews(root)

    # Build vocabulary from FULL training split (no leakage: only token counts)
    vocab = build_vocab(train_texts, max_size=vocab_size)

    # Safe split of training indices
    np.random.seed(seed)
    indices = np.random.permutation(len(train_texts))
    public_idx  = indices[:n_public]
    private_idx = indices[n_public:]

    def _subset(idx_list, texts, labels):
        t = [texts[i]  for i in idx_list]
        l = [labels[i] for i in idx_list]
        return AGNewsDataset(t, l, vocab, seq_len)

    public_set  = _subset(public_idx,  train_texts, train_labels)
    private_set = _subset(private_idx, train_texts, train_labels)
    test_set    = AGNewsDataset(test_texts, test_labels, vocab, seq_len)

    print(f"[AG News Safe Split] Report:")
    print(f"  Vocab size:       {len(vocab) + 2} (incl. PAD/UNK)")
    print(f"  Public (Server):  {len(public_set)} samples (from Train)")
    print(f"  Private (Users):  {len(private_set)} samples (from Train)")
    print(f"  Test (Evaluation):{len(test_set)} samples (from Test)")

    return private_set, public_set, test_set
