#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from pickme_vision.attacks import PGDConfig, PickMePPConfig, pickme_attack, pickmepp_attack, random_attack
from pickme_vision.data import (
    PoolData,
    TensorPoolDataset,
    apply_poison,
    choose_attacker_indices,
    materialize_candidate_pool,
    normalize_dataset_name,
    supported_dataset_names,
)
from pickme_vision.engine import TrainConfig, compute_entropies, evaluate_accuracy, select_topk, selection_rate, train_model
from pickme_vision.models import build_model
from pickme_vision.utils import count_parameters, ensure_dir, save_json, set_seed, resolve_device


@dataclass
class RunSummary:
    dataset: str
    model: str
    attack_mode: str
    seed: int
    candidate_size: int
    attack_size: int
    topk: int
    proxy_epochs: int
    victim_epochs: int
    selection_rate: float
    num_selected_attackers: int
    attacker_entropy_mean: float
    non_attacker_entropy_mean: float
    victim_accuracy: float
    model_params: int
    results_dir: str



def parse_args() -> argparse.Namespace:
    dataset_help = ", ".join(supported_dataset_names())
    parser = argparse.ArgumentParser(
        description="Vision uncertainty-selection attacks on MNIST, CIFAR-10, Fashion-MNIST, and fake data."
    )
    parser.add_argument("--dataset", type=str, default="mnist", help=f"Dataset name or alias. Supported: {dataset_help}")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--model", type=str, default="simplecnn", choices=["simplecnn", "resnet_gn", "depthwisecnn", "tinyvit"])
    parser.add_argument("--attack-mode", type=str, default="random", choices=["random", "pickme", "pickme++"])
    parser.add_argument("--candidate-size", type=int, default=5000)
    parser.add_argument("--attack-size", type=int, default=100)
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--proxy-epochs", type=int, default=3)
    parser.add_argument("--victim-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--pgd-steps", type=int, default=40)
    parser.add_argument("--pickmepp-outer-steps", type=int, default=8)
    parser.add_argument("--pickmepp-inner-steps", type=int, default=5)
    parser.add_argument("--pickmepp-inner-batch-size", type=int, default=128)
    parser.add_argument("--pickmepp-inner-lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--save-poisoned-pool", action="store_true")
    args = parser.parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    return args



def build_train_config(args: argparse.Namespace, epochs: int) -> TrainConfig:
    return TrainConfig(
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
    )



def write_summary_csv(csv_path: Path, summary: RunSummary) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(summary)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)



def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    victim_epochs = args.proxy_epochs if args.victim_epochs is None else args.victim_epochs

    output_root = ensure_dir(args.output_dir)
    run_name = (
        f"{args.dataset}_{args.model}_{args.attack_mode}"
        f"_N{args.candidate_size}_M{args.attack_size}_K{args.topk}_seed{args.seed}"
    )
    run_dir = ensure_dir(output_root / run_name)

    print(f"[INFO] device          : {device}")
    print(f"[INFO] dataset         : {args.dataset}")
    print(f"[INFO] model           : {args.model}")
    print(f"[INFO] attack_mode     : {args.attack_mode}")
    print(f"[INFO] results dir     : {run_dir}")

    pool: PoolData = materialize_candidate_pool(
        dataset_name=args.dataset,
        root=args.data_root,
        candidate_size=args.candidate_size,
        seed=args.seed,
        train=True,
    )

    if args.attack_size >= pool.images.size(0):
        raise ValueError("attack_size must be smaller than candidate_size")

    attacker_indices = choose_attacker_indices(pool.images.size(0), args.attack_size, seed=args.seed + 1)
    attacker_images = pool.images[attacker_indices].clone()
    clean_dataset = TensorPoolDataset(pool.images, pool.labels)

    model_ctor = lambda: build_model(args.model, num_classes=pool.num_classes)
    model_for_proxy = model_ctor().to(device)
    model_params = count_parameters(model_for_proxy)
    print(f"[INFO] trainable params: {model_params:,}")

    if args.attack_mode == "random":
        attacked_images = random_attack(attacker_images)
    elif args.attack_mode == "pickme":
        proxy_config = build_train_config(args, epochs=args.proxy_epochs)
        print("[INFO] training surrogate/proxy model for PickMe...")
        train_model(model_for_proxy, clean_dataset, device=device, config=proxy_config, desc="proxy")
        pgd_config = PGDConfig(
            epsilon=args.epsilon,
            alpha=args.alpha,
            steps=args.pgd_steps,
            batch_size=args.batch_size,
        )
        attacked_images = pickme_attack(model_for_proxy, attacker_images, device=device, config=pgd_config)
    elif args.attack_mode == "pickme++":
        pp_config = PickMePPConfig(
            epsilon=args.epsilon,
            alpha=args.alpha,
            outer_steps=args.pickmepp_outer_steps,
            inner_steps=args.pickmepp_inner_steps,
            inner_batch_size=args.pickmepp_inner_batch_size,
            inner_lr=args.pickmepp_inner_lr,
            model_seed=args.seed + 99,
        )
        attacked_images = pickmepp_attack(
            model_ctor=model_ctor,
            clean_images=pool.images,
            labels=pool.labels,
            attacker_indices=attacker_indices,
            device=device,
            config=pp_config,
        )
    else:
        raise ValueError(f"Unknown attack mode: {args.attack_mode}")

    poisoned_images = apply_poison(pool.images, attacker_indices, attacked_images)
    poisoned_dataset = TensorPoolDataset(poisoned_images, pool.labels)

    if args.save_poisoned_pool:
        torch.save(
            {
                "images": poisoned_images,
                "labels": pool.labels,
                "attacker_indices": attacker_indices,
            },
            run_dir / "poisoned_pool.pt",
        )

    print("[INFO] training victim selector model on poisoned candidate pool...")
    victim_model = model_ctor().to(device)
    victim_config = build_train_config(args, epochs=victim_epochs)
    train_model(victim_model, poisoned_dataset, device=device, config=victim_config, desc="victim")

    print("[INFO] computing entropy scores and selection metrics...")
    entropies = compute_entropies(victim_model, poisoned_dataset, device=device, batch_size=args.batch_size)
    selected_indices = select_topk(entropies, args.topk)
    sr = selection_rate(selected_indices, attacker_indices)
    selected_attackers = len(set(selected_indices.tolist()).intersection(set(attacker_indices.tolist())))

    attacker_mask = torch.zeros_like(entropies, dtype=torch.bool)
    attacker_mask[attacker_indices] = True
    attacker_entropy_mean = float(entropies[attacker_mask].mean().item())
    non_attacker_entropy_mean = float(entropies[~attacker_mask].mean().item())
    victim_acc = evaluate_accuracy(victim_model, poisoned_dataset, device=device, batch_size=args.batch_size)

    summary = RunSummary(
        dataset=args.dataset,
        model=args.model,
        attack_mode=args.attack_mode,
        seed=args.seed,
        candidate_size=int(pool.images.size(0)),
        attack_size=int(attacker_indices.numel()),
        topk=min(args.topk, int(pool.images.size(0))),
        proxy_epochs=args.proxy_epochs,
        victim_epochs=victim_epochs,
        selection_rate=float(sr),
        num_selected_attackers=int(selected_attackers),
        attacker_entropy_mean=float(attacker_entropy_mean),
        non_attacker_entropy_mean=float(non_attacker_entropy_mean),
        victim_accuracy=float(victim_acc),
        model_params=int(model_params),
        results_dir=str(run_dir),
    )

    save_json(run_dir / "summary.json", asdict(summary))
    save_json(
        run_dir / "artifacts.json",
        {
            "attacker_indices": attacker_indices.tolist(),
            "selected_indices": selected_indices.tolist(),
            "entropies": entropies.tolist(),
            "config": vars(args),
        },
    )
    write_summary_csv(output_root / "results_summary.csv", summary)

    print("\n===== FINAL RESULTS =====")
    print(f"dataset                 : {summary.dataset}")
    print(f"model                   : {summary.model}")
    print(f"attack mode             : {summary.attack_mode}")
    print(f"selection rate          : {summary.selection_rate:.4f}")
    print(f"selected attacker count : {summary.num_selected_attackers}/{summary.attack_size}")
    print(f"attacker entropy mean   : {summary.attacker_entropy_mean:.4f}")
    print(f"non-attacker entropy    : {summary.non_attacker_entropy_mean:.4f}")
    print(f"victim train accuracy   : {summary.victim_accuracy:.4f}")
    print(f"saved to                : {run_dir}")


if __name__ == "__main__":
    main()
