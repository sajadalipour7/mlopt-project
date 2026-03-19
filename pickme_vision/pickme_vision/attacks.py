from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from .utils import entropy_from_logits, set_seed

try:
    from torch.func import functional_call
except Exception:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call  # type: ignore


@dataclass
class PGDConfig:
    epsilon: float = 0.3
    alpha: float = 0.01
    steps: int = 40
    batch_size: int = 64


@dataclass
class PickMePPConfig:
    epsilon: float = 0.3
    alpha: float = 0.01
    outer_steps: int = 10
    inner_steps: int = 5
    inner_batch_size: int = 128
    inner_lr: float = 0.05
    model_seed: int = 1234



def random_attack(attacker_images: torch.Tensor) -> torch.Tensor:
    return attacker_images.clone()



def pickme_attack(
    model: nn.Module,
    attacker_images: torch.Tensor,
    device: torch.device,
    config: PGDConfig,
) -> torch.Tensor:
    model.eval()
    outputs = []
    for start in tqdm(range(0, attacker_images.size(0), config.batch_size), desc="pickme", leave=False):
        end = min(start + config.batch_size, attacker_images.size(0))
        x_orig = attacker_images[start:end].to(device)
        x_adv = x_orig.clone().detach()
        for _ in range(config.steps):
            x_adv = x_adv.detach().requires_grad_(True)
            logits = model(x_adv)
            entropy = entropy_from_logits(logits).mean()
            grad = torch.autograd.grad(entropy, x_adv, only_inputs=True)[0]
            with torch.no_grad():
                x_adv = x_adv + config.alpha * grad.sign()
                x_adv = torch.max(torch.min(x_adv, x_orig + config.epsilon), x_orig - config.epsilon)
                x_adv = x_adv.clamp(0.0, 1.0)
        outputs.append(x_adv.detach().cpu())
    return torch.cat(outputs, dim=0)



def _named_params(model: nn.Module, device: torch.device) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, p.detach().to(device).clone().requires_grad_(True)) for name, p in model.named_parameters())



def _named_buffers(model: nn.Module, device: torch.device) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, b.detach().to(device).clone()) for name, b in model.named_buffers())



def _forward_functional(
    model: nn.Module,
    params: OrderedDict[str, torch.Tensor],
    buffers: OrderedDict[str, torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    return functional_call(model, (params, buffers), (x,))



def pickmepp_attack(
    model_ctor,
    clean_images: torch.Tensor,
    labels: torch.Tensor,
    attacker_indices: torch.Tensor,
    device: torch.device,
    config: PickMePPConfig,
) -> torch.Tensor:
    """
    Approximate PickMe++ using differentiable unrolled SGD.

    This attack is intentionally lightweight: each outer step creates a fresh model,
    simulates a small number of inner SGD steps on the poisoned pool, and then
    ascends the attacker images to maximize entropy after those inner updates.
    """
    clean_images = clean_images.to(device)
    labels = labels.to(device)
    attacker_indices = attacker_indices.to(device)
    x_orig = clean_images[attacker_indices].detach().clone()
    x_adv = x_orig.clone().detach()

    outer_bar = tqdm(range(config.outer_steps), desc="pickme++", leave=False)
    for outer_step in outer_bar:
        x_adv = x_adv.detach().clone().requires_grad_(True)
        poisoned_images = clean_images.clone()
        poisoned_images[attacker_indices] = x_adv

        set_seed(config.model_seed)
        inner_model: nn.Module = model_ctor().to(device)
        inner_model.train()
        params = _named_params(inner_model, device)
        buffers = _named_buffers(inner_model, device)

        for _ in range(config.inner_steps):
            batch_idx = torch.randint(
                low=0,
                high=poisoned_images.size(0),
                size=(min(config.inner_batch_size, poisoned_images.size(0)),),
                device=device,
            )
            xb = poisoned_images[batch_idx]
            yb = labels[batch_idx]
            logits = _forward_functional(inner_model, params, buffers, xb)
            loss = F.cross_entropy(logits, yb)
            grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=True)
            params = OrderedDict(
                (name, param - config.inner_lr * grad) for (name, param), grad in zip(params.items(), grads)
            )

        logits_adv = _forward_functional(inner_model, params, buffers, x_adv)
        entropy = entropy_from_logits(logits_adv).mean()
        grad_x = torch.autograd.grad(entropy, x_adv, only_inputs=True)[0]

        with torch.no_grad():
            x_adv = x_adv + config.alpha * grad_x.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + config.epsilon), x_orig - config.epsilon)
            x_adv = x_adv.clamp(0.0, 1.0)

        outer_bar.set_postfix(entropy=f"{float(entropy.detach().cpu().item()):.4f}")

    return x_adv.detach().cpu()
