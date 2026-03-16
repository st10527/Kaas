"""
Dataset Loading for KaaS-Edge

Supports:
- CIFAR-100: Private data for local training (100 classes, 32x32)
- Synthetic: For testing without downloading
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Any

# Check for torch availability
try:
    import torch
    from torch.utils.data import Dataset, DataLoader, Subset
    import torchvision
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # Placeholder
    DataLoader = None


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    name: str
    n_samples: int
    n_classes: int
    image_size: Tuple[int, int, int]  # (C, H, W)
    split: str  # 'train', 'test', 'unlabeled'


class CIFAR100Wrapper:
    """
    Wrapper for CIFAR-100 dataset.
    
    CIFAR-100:
    - 100 classes, 600 images per class
    - 50,000 training images, 10,000 test images
    - 32x32 RGB images
    """
    
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        normalize: bool = True
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for CIFAR-100 loading")
        
        self.root = root
        self.train = train
        
        # Define transforms
        transform_list = [transforms.ToTensor()]
        
        if normalize:
            # CIFAR-100 normalization values
            transform_list.append(
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                )
            )
        
        self.transform = transforms.Compose(transform_list)
        
        # Load dataset
        self.dataset = torchvision.datasets.CIFAR100(
            root=str(self.root),
            train=train,
            download=download,
            transform=self.transform
        )
        self.data = self.dataset.data  # numpy array (N, 32, 32, 3)
        self.targets = np.array(self.dataset.targets)
    
    @property
    def n_samples(self) -> int:
        return len(self.dataset)
    
    @property
    def n_classes(self) -> int:
        return 100
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="CIFAR-100",
            n_samples=self.n_samples,
            n_classes=self.n_classes,
            image_size=(3, 32, 32),
            split="train" if self.train else "test"
        )
    
    def get_class_indices(self) -> Dict[int, List[int]]:
        """Get indices for each class."""
        class_indices = {c: [] for c in range(self.n_classes)}
        for idx, target in enumerate(self.targets):
            class_indices[target].append(idx)
        return class_indices


class STL10Wrapper:
    """
    Wrapper for STL-10 dataset.
    
    STL-10:
    - 10 classes (subset of ImageNet)
    - 5,000 labeled training images
    - 8,000 labeled test images  
    - 100,000 unlabeled images (for semi-supervised learning)
    - 96x96 RGB images
    
    For FD, we use the unlabeled split as public data.
    """
    
    def __init__(
        self,
        root: str = "./data",
        split: str = "unlabeled",
        download: bool = True,
        resize_to: int = 32,
        n_samples: Optional[int] = None,
        seed: int = 42
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for STL-10 loading")
        
        self.root = root
        self.split = split
        self.resize_to = resize_to
        
        # Define transforms
        transform_list = [
            transforms.Resize(resize_to),
            transforms.ToTensor(),
            # ImageNet normalization (STL-10 is from ImageNet)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        
        self.transform = transforms.Compose(transform_list)
        
        # Load dataset
        self.dataset = torchvision.datasets.STL10(
            root=str(self.root),
            split=split,
            download=download,
            transform=self.transform
        )
        
        # Subsample if requested
        if n_samples is not None and n_samples < len(self.dataset):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(self.dataset), n_samples, replace=False)
            self.dataset = Subset(self.dataset, indices)
            self._n_samples = n_samples
        else:
            self._n_samples = len(self.dataset)
    
    @property
    def n_samples(self) -> int:
        return self._n_samples
    
    @property
    def n_classes(self) -> int:
        return 10 if self.split != "unlabeled" else 0
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="STL-10",
            n_samples=self.n_samples,
            n_classes=self.n_classes,
            image_size=(3, self.resize_to, self.resize_to),
            split=self.split
        )


def load_cifar100(
    root: str = "./data",
    download: bool = True,
    normalize: bool = True
) -> Tuple[CIFAR100Wrapper, CIFAR100Wrapper]:
    """
    Load CIFAR-100 train and test sets.
    
    Returns:
        (train_dataset, test_dataset)
    """
    train_data = CIFAR100Wrapper(root, train=True, download=download, normalize=normalize)
    test_data = CIFAR100Wrapper(root, train=False, download=download, normalize=normalize)
    
    print(f"Loaded CIFAR-100:")
    print(f"  Train: {train_data.n_samples} samples")
    print(f"  Test: {test_data.n_samples} samples")
    
    return train_data, test_data


class CIFAR10Wrapper:
    """Wrapper for CIFAR-10 dataset (10 classes)."""
    
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        normalize: bool = True
    ):
        self.root = root
        self.train = train
        
        if normalize:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            self.transform = transforms.ToTensor()
        
        self.dataset = torchvision.datasets.CIFAR10(
            root=str(self.root),
            train=train,
            download=download,
            transform=self.transform
        )
        self.data = self.dataset.data
        self.targets = np.array(self.dataset.targets)
    
    @property
    def n_samples(self) -> int:
        return len(self.dataset)
    
    @property
    def n_classes(self) -> int:
        return 10
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def load_cifar10(
    root: str = "./data",
    download: bool = True,
    normalize: bool = True
) -> Tuple[CIFAR10Wrapper, CIFAR10Wrapper]:
    """
    Load CIFAR-10 train and test sets.
    
    Returns:
        (train_dataset, test_dataset)
    """
    train_data = CIFAR10Wrapper(root, train=True, download=download, normalize=normalize)
    test_data = CIFAR10Wrapper(root, train=False, download=download, normalize=normalize)
    
    print(f"Loaded CIFAR-10:")
    print(f"  Train: {train_data.n_samples} samples")
    print(f"  Test: {test_data.n_samples} samples")
    
    return train_data, test_data


def load_dataset(
    name: str = "cifar100",
    root: str = "./data",
    download: bool = True,
    normalize: bool = True
):
    """
    Unified dataset loader.
    
    Args:
        name: 'cifar10' or 'cifar100'
        root: Data directory
        download: Whether to download
        normalize: Whether to normalize
        
    Returns:
        (train_dataset, test_dataset, n_classes)
    """
    if name.lower() == "cifar10":
        train, test = load_cifar10(root, download, normalize)
        return train, test, 10
    elif name.lower() == "cifar100":
        train, test = load_cifar100(root, download, normalize)
        return train, test, 100
    elif name.lower() in ("emnist", "emnist_byclass"):
        train, test = load_emnist_byclass(root, download)
        return train, test, 62
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose 'cifar10', 'cifar100', or 'emnist_byclass'")


def load_stl10(
    root: str = "./data",
    split: str = "unlabeled",
    download: bool = True,
    n_samples: int = 5000,
    resize_to: int = 32,
    seed: int = 42
) -> STL10Wrapper:
    """
    Load STL-10 dataset as public data for knowledge distillation.
    
    Args:
        root: Data directory
        split: Dataset split ('unlabeled' recommended for FD)
        download: Whether to download if not present
        n_samples: Number of samples to use (default 5000)
        resize_to: Resize to match CIFAR (default 32)
        seed: Random seed for subsampling
        
    Returns:
        STL10Wrapper instance
    """
    data = STL10Wrapper(
        root=root,
        split=split,
        download=download,
        resize_to=resize_to,
        n_samples=n_samples,
        seed=seed
    )
    
    print(f"Loaded STL-10 ({split}):")
    print(f"  Samples: {data.n_samples}")
    print(f"  Resized to: {resize_to}x{resize_to}")
    
    return data


def get_data_loaders(
    train_dataset: Any,
    test_dataset: Any,
    public_dataset: Any,
    batch_size: int = 64,
    num_workers: int = 2
) -> Dict[str, Any]:
    """
    Create DataLoaders for train, test, and public datasets.
    
    Returns:
        Dictionary with 'train', 'test', 'public' DataLoaders
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for DataLoader")
    
    # Handle wrapped datasets
    train_ds = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
    test_ds = test_dataset.dataset if hasattr(test_dataset, 'dataset') else test_dataset
    public_ds = public_dataset.dataset if hasattr(public_dataset, 'dataset') else public_dataset
    
    loaders = {
        'train': DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'public': DataLoader(
            public_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return loaders


# =========================================================================
# EMNIST-ByClass (JPDC 2026 — naturally non-IID second dataset)
# =========================================================================

class EMNISTByClassWrapper:
    """
    Wrapper for EMNIST ByClass split.

    EMNIST-ByClass:
    - 62 classes  (0-9, A-Z, a-z)
    - 814,255 images (train+test)
    - 28×28 grayscale → resized to ``img_size`` and replicated to 3 channels
      so it can share the same CNN architecture as CIFAR-100.

    For JPDC experiments we use Dirichlet(α=0.1) partitioning to mimic
    extreme writer-level heterogeneity.  This is simpler and more
    reproducible than parsing LEAF JSON files.
    """

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        img_size: int = 32,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for EMNIST loading")

        self.root = root
        self.train = train
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1ch → 3ch
        ])

        self.dataset = torchvision.datasets.EMNIST(
            root=str(self.root),
            split='byclass',
            train=train,
            download=download,
            transform=self.transform,
        )
        self.targets = np.array(self.dataset.targets)

    @property
    def n_samples(self) -> int:
        return len(self.dataset)

    @property
    def n_classes(self) -> int:
        return 62

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="EMNIST-ByClass",
            n_samples=self.n_samples,
            n_classes=self.n_classes,
            image_size=(3, self.img_size, self.img_size),
            split="train" if self.train else "test",
        )

    def get_class_indices(self) -> Dict[int, List[int]]:
        class_indices: Dict[int, List[int]] = {c: [] for c in range(self.n_classes)}
        for idx, target in enumerate(self.targets):
            class_indices[int(target)].append(idx)
        return class_indices


def load_emnist_byclass(
    root: str = "./data",
    download: bool = True,
    img_size: int = 32,
) -> Tuple[EMNISTByClassWrapper, EMNISTByClassWrapper]:
    """
    Load EMNIST-ByClass train and test sets.

    Returns (train_dataset, test_dataset).
    """
    train = EMNISTByClassWrapper(root, train=True, download=download, img_size=img_size)
    test = EMNISTByClassWrapper(root, train=False, download=download, img_size=img_size)
    print(f"Loaded EMNIST-ByClass:")
    print(f"  Train: {train.n_samples} samples, {train.n_classes} classes")
    print(f"  Test:  {test.n_samples} samples, {test.n_classes} classes")
    return train, test


def load_emnist_safe_split(
    root: str = "./data",
    n_public: int = 10000,
    seed: int = 42,
    img_size: int = 32,
):
    """
    Safe split for EMNIST-ByClass (mirrors load_cifar100_safe_split).

    Splits the training set into:
      - private_set (for local training, with augmentation)
      - public_set  (for distillation, without augmentation)
    Keeps the test set untouched.
    """
    import torch
    from torch.utils.data import Subset

    stats = ((0.1307,), (0.3081,))

    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])

    full_train_aug = torchvision.datasets.EMNIST(
        root=root, split='byclass', train=True, download=True,
        transform=transform_train,
    )
    full_train_clean = torchvision.datasets.EMNIST(
        root=root, split='byclass', train=True, download=True,
        transform=transform_test,
    )
    test_set = torchvision.datasets.EMNIST(
        root=root, split='byclass', train=False, download=True,
        transform=transform_test,
    )

    np.random.seed(seed)
    indices = np.random.permutation(len(full_train_aug))
    public_indices = indices[:n_public]
    private_indices = indices[n_public:]

    private_set = Subset(full_train_aug, private_indices)
    public_set = Subset(full_train_clean, public_indices)

    print(f"[EMNIST Safe Split]")
    print(f"  Public:  {len(public_set)} samples")
    print(f"  Private: {len(private_set)} samples")
    print(f"  Test:    {len(test_set)} samples")

    return private_set, public_set, test_set


class SyntheticDataset:
    """
    Synthetic dataset for testing without downloading real data.
    
    Useful for unit tests and quick experiments.
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        n_classes: int = 100,
        image_size: Tuple[int, int, int] = (3, 32, 32),
        seed: int = 42
    ):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.image_size = image_size
        
        rng = np.random.RandomState(seed)
        
        # Generate random data
        self.data = rng.randn(n_samples, *image_size).astype(np.float32)
        self.targets = rng.randint(0, n_classes, n_samples)
        
        if TORCH_AVAILABLE:
            self._data_tensor = torch.from_numpy(self.data)
            self._targets_tensor = torch.from_numpy(self.targets).long()
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if TORCH_AVAILABLE:
            return self._data_tensor[idx], self._targets_tensor[idx]
        return self.data[idx], self.targets[idx]
    
    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="Synthetic",
            n_samples=self.n_samples,
            n_classes=self.n_classes,
            image_size=self.image_size,
            split="synthetic"
        )
    
    def get_class_indices(self) -> Dict[int, List[int]]:
        """Get indices for each class."""
        class_indices = {c: [] for c in range(self.n_classes)}
        for idx, target in enumerate(self.targets):
            class_indices[int(target)].append(idx)
        return class_indices


def create_synthetic_datasets(
    n_train: int = 5000,
    n_test: int = 1000,
    n_public: int = 2000,
    n_classes: int = 100,
    seed: int = 42
) -> Tuple[SyntheticDataset, SyntheticDataset, SyntheticDataset]:
    """
    Create synthetic datasets for testing.
    
    Returns:
        (train_dataset, test_dataset, public_dataset)
    """
    train = SyntheticDataset(n_train, n_classes, seed=seed)
    test = SyntheticDataset(n_test, n_classes, seed=seed+1)
    public = SyntheticDataset(n_public, n_classes=10, seed=seed+2)  # STL-10 has 10 classes
    
    print(f"Created synthetic datasets:")
    print(f"  Train: {n_train} samples, {n_classes} classes")
    print(f"  Test: {n_test} samples, {n_classes} classes")
    print(f"  Public: {n_public} samples, 10 classes")
    
    return train, test, public

# ==========================================
# Safe Split: CIFAR-100 train/public/test
# ==========================================
def load_cifar100_safe_split(root='./data', n_public=10000, seed=42):
    """
    Safe Split:
    將 CIFAR-100 Training Set (50k) 切割為：
    1. Private Data (40k): 給 User 做 Local Training (Non-IID)
    2. Public Data (10k): 給 Server 做 Knowledge Distillation (IID)
    
    保留原本的 Test Set (10k) 僅作最終評估，確保無洩漏 (Zero Leakage)。
    """
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import Subset
    import numpy as np

    # 1. 定義轉換
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # 2. 載入完整 Training Set (with augmentation for private data)
    full_train_aug = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform_train
    )
    # Also load WITHOUT augmentation (for public data inference)
    full_train_clean = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform_test
    )
    
    # 3. 載入 Test Set (神聖不可侵犯)
    test_set = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True, transform=transform_test
    )

    # 4. 執行安全切割 (Safe Split)
    # 使用固定 seed 確保每次實驗的 Public Data 是一樣的
    np.random.seed(seed)
    indices = np.random.permutation(len(full_train_aug))
    
    public_indices = indices[:n_public]      # 前 n_public 給 Public
    private_indices = indices[n_public:]     # 後面給 Private
    
    # Private data: uses augmentation (RandomCrop + Flip) for local training
    private_set = Subset(full_train_aug, private_indices)
    # Public data: NO augmentation (deterministic inference for logit alignment)
    public_set = Subset(full_train_clean, public_indices)

    print(f"[Safe Split] Report:")
    print(f"  Public (Server):  {len(public_set)} samples (from Train)")
    print(f"  Private (Users):  {len(private_set)} samples (from Train)")
    print(f"  Test (Evaluation):{len(test_set)} samples (from Test)")
    
    return private_set, public_set, test_set