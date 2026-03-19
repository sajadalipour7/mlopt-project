#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-mnist}
OUTPUT_DIR=${2:-./outputs_grid}
DEVICE=${3:-auto}

MODELS=(simplecnn resnet_gn depthwisecnn tinyvit)
ATTACKS=(random pickme)

for model in "${MODELS[@]}"; do
  for attack in "${ATTACKS[@]}"; do
    python run_attack.py \
      --dataset "$DATASET" \
      --model "$model" \
      --attack-mode "$attack" \
      --candidate-size 5000 \
      --attack-size 100 \
      --topk 500 \
      --proxy-epochs 3 \
      --batch-size 128 \
      --device "$DEVICE" \
      --output-dir "$OUTPUT_DIR"
  done
done

# Smaller PickMe++ runs
# tinyvit removed here because that variant crashes
for model in simplecnn resnet_gn depthwisecnn; do
  python run_attack.py \
    --dataset "$DATASET" \
    --model "$model" \
    --attack-mode pickme++ \
    --candidate-size 5000 \
    --attack-size 100 \
    --topk 500 \
    --proxy-epochs 3 \
    --batch-size 128 \
    --epsilon 0.3 \
    --alpha 0.01 \
    --pickmepp-outer-steps 6 \
    --pickmepp-inner-steps 4 \
    --pickmepp-inner-batch-size 128 \
    --pickmepp-inner-lr 0.05 \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR"
done