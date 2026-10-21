#!/bin/bash
# Verify every model in the matrix: arch, quant, size, load/generate timing
# and a seed prompt. Writes one JSON per model into models/verify/.
cd /Users/charlie/Documents/mlx-lab || exit 1
MODELS_DIR="$HOME/MLXLoRAStudio/models"
NAMES=("LFM2.5-24B-A2B-4bit" "gemma-4-12B-4bit" "gemma-4-12B-it-4bit" "Qwen3.5-9B-DefiantFable-4bit")
PROMPTS=(
  "The capital of France is"
  "Explain what a mixture-of-experts model is in one sentence."
  "List three uses of a 24B MoE language model on a laptop."
  "Write a one-sentence summary of a small expert-routed model."
)
mkdir -p models/verify
for i in "${!NAMES[@]}"; do
  n="${NAMES[$i]}"
  echo "=== [$(date +%H:%M:%S)] verify $n"
  .venv/bin/python scripts/verify_model.py "$MODELS_DIR/$n" "${PROMPTS[$i]}" \
    > "models/verify/$n.json" 2> "models/verify/$n.err"
  echo "    exit=$? -> models/verify/$n.json"
done
echo "=== [$(date +%H:%M:%S)] VERIFY ALL DONE ==="
