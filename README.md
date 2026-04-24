# mlopt-project

## Pick Me: Fooling Uncertainty-Based Data Selection

This repository contains the implementation for a project studying adversarial attacks on uncertainty-based data selection (coreset selection) pipelines.

## Presentation Video
https://rpi.box.com/s/rnu2xjl971l195ur98x89be6u8ya4suz

## Overview

Modern ML training pipelines often select a small, high-value subset from a large data pool using predictive entropy as a proxy for "informativeness." This project shows that an adversary controlling a small fraction of the candidate data can craft inputs that maximize their entropy under a proxy model, causing them to be disproportionately selected into the final training coreset.

Two attacks are implemented:

- **Pick Me** — directly maximizes predictive entropy via projected gradient descent (PGD) on a surrogate proxy model.
- **Pick Me++** — a bilevel optimization extension that accounts for the victim's retraining dynamics, ensuring adversarial samples remain high-entropy even after the model is updated.

## Repository Structure

```
main/
  attack.py       # Pick Me (PGD entropy-maximization attack)
  model.py        # SimpleNN (MLP) and ResNet18 architectures
  train.py        # Model training and evaluation utilities
  main.py         # Main experiment script
  script.sh       # Example run commands
  bilevel/
    bilevel.ipynb # Pick Me++ bilevel optimization (notebook)
```

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy, matplotlib, tqdm
- [higher](https://github.com/facebookresearch/higher) (for Pick Me++ bilevel optimization)

## Usage

Run an experiment with `main/main.py`:

```bash
python main.py \
  --dataset MNIST \           # MNIST | FashionMNIST | CIFAR10
  --proxy_model SimpleNN \    # SimpleNN | ResNet
  --victim_model SimpleNN \   # SimpleNN | ResNet
  --total_dataset_size 60000 \
  --poison_dataset_size 100 \
  --proxy_epochs 3 \
  --victim_epochs 3 \
  --attack_name pickme
```

See [main/script.sh](main/script.sh) for additional example commands.

For **Pick Me++**, open and run [main/bilevel/bilevel.ipynb](main/bilevel/bilevel.ipynb).

## Key Arguments

| Argument | Description |
|---|---|
| `--dataset` | Dataset: `MNIST`, `FashionMNIST`, or `CIFAR10` |
| `--proxy_model` | Attacker's proxy model: `SimpleNN` or `ResNet` |
| `--victim_model` | Victim's proxy model: `SimpleNN` or `ResNet` |
| `--total_dataset_size` | Size of the full data pool $\|D\|$ |
| `--poison_dataset_size` | Number of attacker-controlled samples $\|M\|$ |
| `--proxy_epochs` | Training epochs for the attacker's proxy model $E_{proxy}$ |
| `--victim_epochs` | Training epochs for the victim's proxy model $E_{victim}$ |
| `--attack_name` | Attack to run (`pickme`) |
