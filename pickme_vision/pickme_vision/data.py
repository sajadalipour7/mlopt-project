from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class PoolData:
    images: torch.Tensor
    labels: torch.Tensor
    dataset_name: str
    num_classes: int


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    torchvision_name: str | None
    num_classes: int
    color_mode: str
    aliases: tuple[str, ...] = ()


class TensorPoolDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        assert images.size(0) == labels.size(0)
        self.images = images.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


class FakeVisionDataset(Dataset):
    def __init__(self, size: int = 1000, num_classes: int = 10, image_size: int = 32, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.images = torch.rand(size, 3, image_size, image_size, generator=g)
        self.labels = torch.randint(0, num_classes, (size,), generator=g)

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def _fake_loader(*, train: bool, root: str | Path) -> Dataset:
    del root
    return FakeVisionDataset(size=2000 if train else 500)


DATASET_SPECS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="mnist",
        torchvision_name="MNIST",
        num_classes=10,
        color_mode="grayscale",
        aliases=("minist",),
    ),
    "cifar10": DatasetSpec(
        name="cifar10",
        torchvision_name="CIFAR10",
        num_classes=10,
        color_mode="rgb",
        aliases=("cifar-10",),
    ),
    "fashionmnist": DatasetSpec(
        name="fashionmnist",
        torchvision_name="FashionMNIST",
        num_classes=10,
        color_mode="grayscale",
        aliases=("fashion-mnist", "fmnist"),
    ),
    "fake": DatasetSpec(
        name="fake",
        torchvision_name=None,
        num_classes=10,
        color_mode="rgb",
    ),
}

DATASET_ALIASES: dict[str, str] = {
    alias: spec.name
    for spec in DATASET_SPECS.values()
    for alias in (spec.name, *spec.aliases)
}


def supported_dataset_names() -> tuple[str, ...]:
    return tuple(DATASET_SPECS)


def normalize_dataset_name(name: str) -> str:
    key = name.lower()
    if key not in DATASET_ALIASES:
        supported = ", ".join(sorted(DATASET_ALIASES))
        raise ValueError(f"dataset must be one of: {supported}")
    return DATASET_ALIASES[key]


def dataset_spec(name: str) -> DatasetSpec:
    return DATASET_SPECS[normalize_dataset_name(name)]


def _pil_to_tensor(img: Image.Image, dataset_name: str) -> torch.Tensor:
    spec = dataset_spec(dataset_name)
    if spec.color_mode == "grayscale":
        img = img.convert("L").resize((32, 32), Image.BILINEAR).convert("RGB")
    else:
        img = img.convert("RGB").resize((32, 32), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor



def load_base_dataset(name: str, root: str | Path, train: bool = True) -> Dataset:
    spec = dataset_spec(name)
    if spec.name == "fake":
        return _fake_loader(root=root, train=train)

    try:
        from torchvision import datasets
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torchvision could not be imported. Please install matching torch/torchvision versions. "
            f"Original error: {e}"
        ) from e

    dataset_cls = getattr(datasets, spec.torchvision_name)
    return dataset_cls(root=str(root), train=train, transform=None, download=True)



def materialize_candidate_pool(
    dataset_name: str,
    root: str | Path,
    candidate_size: int,
    seed: int,
    train: bool = True,
) -> PoolData:
    spec = dataset_spec(dataset_name)
    dataset = load_base_dataset(spec.name, root=root, train=train)
    total_size = len(dataset)
    if candidate_size <= 0 or candidate_size > total_size:
        candidate_size = total_size

    generator = torch.Generator().manual_seed(seed)
    subset_indices = torch.randperm(total_size, generator=generator)[:candidate_size]

    images = []
    labels = []
    for idx in subset_indices.tolist():
        img, label = dataset[idx]
        if isinstance(img, torch.Tensor):
            tensor = img.float().clamp(0.0, 1.0)
        else:
            tensor = _pil_to_tensor(img, spec.name)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        elif tensor.size(0) == 1:
            tensor = tensor.repeat(3, 1, 1)
        images.append(tensor)
        labels.append(int(label))

    images_tensor = torch.stack(images).float().clamp(0.0, 1.0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return PoolData(images=images_tensor, labels=labels_tensor, dataset_name=spec.name, num_classes=spec.num_classes)



def choose_attacker_indices(num_samples: int, attack_size: int, seed: int) -> torch.Tensor:
    if attack_size <= 0:
        raise ValueError("attack_size must be positive")
    if attack_size >= num_samples:
        raise ValueError("attack_size must be smaller than candidate pool size")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_samples, generator=generator)[:attack_size]



def apply_poison(images: torch.Tensor, attacker_indices: torch.Tensor, attacked_images: torch.Tensor) -> torch.Tensor:
    poisoned = images.clone()
    poisoned[attacker_indices] = attacked_images.detach().cpu()
    return poisoned.clamp(0.0, 1.0)
