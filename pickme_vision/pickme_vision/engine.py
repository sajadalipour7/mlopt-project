from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .utils import entropy_from_logits


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"



def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    key = config.optimizer.lower()
    if key == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if key == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    raise ValueError("optimizer must be one of: adam, adamw, sgd")



def train_model(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    config: TrainConfig,
    desc: str = "train",
) -> None:
    model.to(device)
    model.train()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    optimizer = build_optimizer(model, config)

    epoch_bar = tqdm(range(config.epochs), desc=desc, leave=False)
    for epoch in epoch_bar:
        running_loss = 0.0
        running_correct = 0
        total = 0
        batch_bar = tqdm(loader, desc=f"{desc} epoch {epoch + 1}/{config.epochs}", leave=False)
        for images, labels in batch_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)
            running_correct += int((preds == labels).sum().item())
            total += labels.size(0)
            batch_bar.set_postfix(loss=f"{running_loss / max(total, 1):.4f}", acc=f"{running_correct / max(total, 1):.4f}")

        epoch_bar.set_postfix(loss=f"{running_loss / max(total, 1):.4f}", acc=f"{running_correct / max(total, 1):.4f}")


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int = 256) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def compute_entropies(model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs = []
    for images, _ in tqdm(loader, desc="entropy", leave=False):
        images = images.to(device)
        logits = model(images)
        outputs.append(entropy_from_logits(logits).cpu())
    return torch.cat(outputs, dim=0)



def select_topk(entropies: torch.Tensor, topk: int) -> torch.Tensor:
    topk = min(topk, entropies.numel())
    return torch.topk(entropies, k=topk, largest=True).indices



def selection_rate(selected_indices: torch.Tensor, attacker_indices: torch.Tensor) -> float:
    selected_set = set(selected_indices.detach().cpu().tolist())
    attacker_set = set(attacker_indices.detach().cpu().tolist())
    hits = len(selected_set.intersection(attacker_set))
    return hits / max(len(attacker_set), 1)
