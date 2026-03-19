from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


EPS = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)



def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(EPS))).sum(dim=1)



def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out



def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    serializable: Dict[str, Any] = {}
    for key, value in payload.items():
        if is_dataclass(value):
            serializable[key] = asdict(value)
        elif isinstance(value, Path):
            serializable[key] = str(value)
        elif isinstance(value, torch.Tensor):
            serializable[key] = value.detach().cpu().tolist()
        else:
            serializable[key] = value
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)



def tensor_to_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)



def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
