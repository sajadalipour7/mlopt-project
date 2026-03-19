#!/usr/bin/env bash
set -euo pipefail

python run_attack.py \
  --dataset fake \
  --model simplecnn \
  --attack-mode random \
  --candidate-size 256 \
  --attack-size 16 \
  --topk 32 \
  --proxy-epochs 1 \
  --batch-size 32 \
  --device auto \
  --output-dir ./outputs_smoke

python run_attack.py \
  --dataset fake \
  --model simplecnn \
  --attack-mode pickme \
  --candidate-size 256 \
  --attack-size 16 \
  --topk 32 \
  --proxy-epochs 1 \
  --batch-size 32 \
  --epsilon 0.1 \
  --alpha 0.02 \
  --pgd-steps 2 \
  --device auto \
  --output-dir ./outputs_smoke

python run_attack.py \
  --dataset fake \
  --model simplecnn \
  --attack-mode pickme++ \
  --candidate-size 128 \
  --attack-size 8 \
  --topk 16 \
  --proxy-epochs 1 \
  --batch-size 32 \
  --epsilon 0.1 \
  --alpha 0.02 \
  --pickmepp-outer-steps 1 \
  --pickmepp-inner-steps 1 \
  --pickmepp-inner-batch-size 32 \
  --pickmepp-inner-lr 0.05 \
  --device auto \
  --output-dir ./outputs_smoke
