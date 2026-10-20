#!/bin/bash
# Fetch the three target models as MLX 4-bit into ~/MLXLoRAStudio/models
set -uo pipefail
DEST="$HOME/MLXLoRAStudio/models"
mkdir -p "$DEST"
fetch () {  # repo_id  local_name
  local repo="$1" name="$2"
  echo "=== [$(date +%H:%M:%S)] fetching $repo -> $name"
  if hf download "$repo" --quiet 2>&1 | tail -2; then :; else echo "FAILED $repo"; return 1; fi
  local snap
  snap=$(hf repo-files "$repo" 2>/dev/null >/dev/null; python3 - "$repo" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], local_files_only=True))
PY
)
  echo "    snapshot: $snap"
  ln -sfn "$snap" "$DEST/$name"
  echo "    linked  : $DEST/$name -> $snap"
}
fetch "mlx-community/LFM2.5-24B-A2B-OptiQ-4bit" "LFM2.5-24B-A2B-4bit"
fetch "mlx-community/gemma-4-12B-4bit" "gemma-4-12B-4bit"
fetch "DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP" "Qwen3.5-9B-DefiantFable-bf16"
echo "=== [$(date +%H:%M:%S)] all downloads done ==="
