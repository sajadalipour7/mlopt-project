# PickMe / PickMe++ vision codebase

This repo implements the three attack modes from the proposal on **MNIST**, **CIFAR-10**, and **Fashion-MNIST**:

- **random**: attacker controls a random subset of samples and does not optimize them.
- **pickme**: attacker maximizes sample entropy under a trained proxy model using PGD.
- **pickme++**: attacker uses a lightweight differentiable bilevel approximation so the samples stay high-entropy **after** inner-loop training.


## Supported datasets

- `mnist`
- `cifar10`
- `fashionmnist`
- `fake` (only for smoke tests / debugging)

Accepted aliases:

- `minist` → `mnist`
- `cifar-10` → `cifar10`
- `fashion-mnist`, `fmnist` → `fashionmnist`

## Supported models

All models are custom 32x32 classifiers and are fully differentiable for PickMe++.

- `simplecnn`
- `resnet_gn`        → small residual network with GroupNorm
- `depthwisecnn`     → lightweight MobileNet-style depthwise separable CNN
- `tinyvit`          → compact vision transformer

These models were chosen so that **all three attacks** can be run in one unified codebase without depending on extra bilevel libraries.

---

## Folder structure

```text
pickme_vision/
├── README.md
├── requirements.txt
├── run_attack.py
├── pickme_vision/
│   ├── __init__.py
│   ├── attacks.py
│   ├── data.py
│   ├── engine.py
│   ├── models.py
│   └── utils.py
└── scripts/
    ├── run_lab_grid.sh
    └── run_smoke_tests.sh
```

---

## Environment setup

### Option A: Conda

```bash
conda create -n pickmevision python=3.10 -y
conda activate pickmevision
pip install -r requirements.txt
```

### Option B: venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If your machine has a GPU and PyTorch CUDA is already installed, the script will automatically use it when you pass `--device auto`.

---

## Main command

```bash
python run_attack.py \
  --dataset mnist \
  --model simplecnn \
  --attack-mode random \
  --candidate-size 5000 \
  --attack-size 100 \
  --topk 500 \
  --proxy-epochs 3 \
  --batch-size 128 \
  --lr 1e-3 \
  --optimizer adamw \
  --device auto \
  --output-dir ./outputs
```

---

## Important idea behind the pipeline

For every run, the script does the following:

1. Load a candidate pool from MNIST, CIFAR-10, or Fashion-MNIST.
2. Choose an attacker-controlled subset `M` of size `attack_size`.
3. Create poisoned attacker samples according to the chosen attack mode.
4. Replace those samples inside the candidate pool.
5. Train the victim selector model on the poisoned pool.
6. Compute predictive entropy on the whole pool.
7. Select top-`k` highest entropy samples.
8. Report the **selection rate** of attacker samples.

The main metric is:

```text
selection_rate = (# attacker samples selected into top-k) / (total attacker samples)
```

---

## Attack modes

### 1. Random baseline

The attacker subset is inserted unchanged.

```bash
python run_attack.py \
  --dataset mnist \
  --model simplecnn \
  --attack-mode random \
  --candidate-size 5000 \
  --attack-size 100 \
  --topk 500 \
  --proxy-epochs 3 \
  --device auto
```

### 2. PickMe

Train a proxy model on the clean candidate pool, then use PGD to maximize entropy of attacker samples.

```bash
python run_attack.py \
  --dataset mnist \
  --model simplecnn \
  --attack-mode pickme \
  --candidate-size 5000 \
  --attack-size 100 \
  --topk 500 \
  --proxy-epochs 3 \
  --epsilon 0.3 \
  --alpha 0.01 \
  --pgd-steps 40 \
  --device auto
```

### 3. PickMe++

This implementation uses a **lightweight differentiable unrolled SGD approximation**:

- outer loop updates attacker samples,
- inner loop unrolls a few training steps of the victim model,
- objective is entropy **after** the inner training steps.

This is the expensive mode. Start small.

```bash
python run_attack.py \
  --dataset mnist \
  --model simplecnn \
  --attack-mode pickme++ \
  --candidate-size 2000 \
  --attack-size 50 \
  --topk 200 \
  --proxy-epochs 3 \
  --epsilon 0.3 \
  --alpha 0.01 \
  --pickmepp-outer-steps 8 \
  --pickmepp-inner-steps 5 \
  --pickmepp-inner-batch-size 128 \
  --pickmepp-inner-lr 0.05 \
  --device auto
```

---

## Recommended 

### A. Quick smoke test (no downloads needed)

```bash
bash scripts/run_smoke_tests.sh
```

This uses the `fake` dataset and tiny settings to verify the code path.

### B. MNIST standard runs

```bash
python run_attack.py --dataset mnist --model simplecnn    --attack-mode random   --candidate-size 5000 --attack-size 100 --topk 500 --proxy-epochs 3 --device auto
python run_attack.py --dataset mnist --model simplecnn    --attack-mode pickme   --candidate-size 5000 --attack-size 100 --topk 500 --proxy-epochs 3 --epsilon 0.3 --alpha 0.01 --pgd-steps 40 --device auto
python run_attack.py --dataset mnist --model simplecnn    --attack-mode pickme++ --candidate-size 2000 --attack-size 50  --topk 200 --proxy-epochs 3 --epsilon 0.3 --alpha 0.01 --pickmepp-outer-steps 8 --pickmepp-inner-steps 5 --device auto
```

### C. CIFAR-10 standard runs

```bash
python run_attack.py --dataset cifar10 --model resnet_gn    --attack-mode random   --candidate-size 5000 --attack-size 100 --topk 500 --proxy-epochs 5 --device auto
python run_attack.py --dataset cifar10 --model resnet_gn    --attack-mode pickme   --candidate-size 5000 --attack-size 100 --topk 500 --proxy-epochs 5 --epsilon 0.1 --alpha 0.005 --pgd-steps 30 --device auto
python run_attack.py --dataset cifar10 --model resnet_gn    --attack-mode pickme++ --candidate-size 2000 --attack-size 50  --topk 200 --proxy-epochs 5 --epsilon 0.1 --alpha 0.005 --pickmepp-outer-steps 6 --pickmepp-inner-steps 4 --device auto
```

### D. Fashion-MNIST starter runs

```bash
python run_attack.py --dataset fashionmnist --model simplecnn --attack-mode random --candidate-size 5000 --attack-size 100 --topk 500 --proxy-epochs 3 --device auto
python run_attack.py --dataset fashionmnist --model simplecnn --attack-mode pickme --candidate-size 5000 --attack-size 100 --topk 500 --proxy-epochs 3 --epsilon 0.3 --alpha 0.01 --pgd-steps 40 --device auto
python run_attack.py --dataset fashion-mnist --model resnet_gn --attack-mode pickme++ --candidate-size 2000 --attack-size 50 --topk 200 --proxy-epochs 3 --epsilon 0.3 --alpha 0.01 --pickmepp-outer-steps 8 --pickmepp-inner-steps 5 --device auto
```

### E. Model-comparison grid on MNIST

```bash
python run_attack.py --dataset mnist --model simplecnn     --attack-mode pickme --candidate-size 5000 --attack-size 100 --topk 500 --proxy-epochs 3 --device auto
python run_attack.py --dataset mnist --model resnet_gn     --attack-mode pickme --candidate-size 5000 --attack-size 100 --topk 500 --proxy-epochs 3 --device auto
python run_attack.py --dataset mnist --model depthwisecnn  --attack-mode pickme --candidate-size 5000 --attack-size 100 --topk 500 --proxy-epochs 3 --device auto
python run_attack.py --dataset mnist --model tinyvit       --attack-mode pickme --candidate-size 5000 --attack-size 100 --topk 500 --proxy-epochs 3 --device auto
```

---

## Output files

Each run creates a folder like:

```text
outputs/mnist_simplecnn_pickme_N5000_M100_K500_seed7/
```

Inside it you will find:

- `summary.json`      → main metrics for the run
- `artifacts.json`    → selected indices, attacker indices, entropy list, config
- `poisoned_pool.pt`  → saved only when `--save-poisoned-pool` is passed

A cumulative table is also appended to:

```text
outputs/results_summary.csv
```

---

## Key arguments

### Dataset / model

- `--dataset` : `mnist`, `cifar10`, `fashionmnist`, `fake`
- `--model`   : `simplecnn`, `resnet_gn`, `depthwisecnn`, `tinyvit`

### Pool setup

- `--candidate-size` : number of samples in the candidate pool
- `--attack-size`    : number of attacker-controlled samples
- `--topk`           : selector keeps the top-k highest entropy samples

### Training

- `--proxy-epochs`   : proxy epochs used for PickMe and as default victim epochs
- `--victim-epochs`  : optional override for the victim model training epochs
- `--batch-size`
- `--lr`
- `--optimizer`      : `adam`, `adamw`, or `sgd`
- `--weight-decay`

### PickMe

- `--epsilon`
- `--alpha`
- `--pgd-steps`

### PickMe++

- `--pickmepp-outer-steps`
- `--pickmepp-inner-steps`
- `--pickmepp-inner-batch-size`
- `--pickmepp-inner-lr`

---

## Practical tips

### 1. Start with MNIST first

MNIST is the fastest sanity check.

### 2. Start PickMe++ with smaller pools

Good first values:

- `candidate-size = 1000` to `2000`
- `attack-size = 20` to `50`
- `outer-steps = 4` to `8`
- `inner-steps = 3` to `5`

### 3. Increase size only after confirming the pipeline works

PickMe++ becomes expensive quickly because it differentiates through inner training.

### 4. Use the same seed when comparing attacks

This keeps the attacker subset and candidate pool comparable.

---

## Example interpretation

Suppose the output says:

```text
selection rate          : 0.4200
selected attacker count : 42/100
```

That means 42 of the 100 attacker-controlled samples were chosen by the entropy selector into the top-k subset.

---

## One-line commands

### Random

```bash
python run_attack.py --dataset mnist --model simplecnn --attack-mode random --candidate-size 5000 --attack-size 100 --topk 500 --proxy-epochs 3 --device auto
```

### PickMe

```bash
python run_attack.py --dataset mnist --model simplecnn --attack-mode pickme --candidate-size 5000 --attack-size 100 --topk 500 --proxy-epochs 3 --epsilon 0.3 --alpha 0.01 --pgd-steps 40 --device auto
```

### PickMe++

```bash
python run_attack.py --dataset mnist --model simplecnn --attack-mode pickme++ --candidate-size 2000 --attack-size 50 --topk 200 --proxy-epochs 3 --epsilon 0.3 --alpha 0.01 --pickmepp-outer-steps 8 --pickmepp-inner-steps 5 --pickmepp-inner-batch-size 128 --pickmepp-inner-lr 0.05 --device auto
```

---

## Notes on the PickMe++ implementation

This is a **working research prototype**, not a full large-scale bilevel system.

What it does:
- unrolls differentiable SGD steps,
- optimizes attacker images so entropy remains high after inner training.

What it does not do yet:
- full-epoch bilevel retraining through the entire dataset,
- large-scale hyperparameter tuning,
- backdoor trigger constraints.
