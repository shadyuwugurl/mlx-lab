#!/bin/bash
# NOTE: HF_HUB_DISABLE_XET=1 is required — the xet transport hangs at 0 KB/s on
# the DavidAU repo (verified: 0 KB/s for 4 min with xet, 13.9 MB/s with plain HTTP).
set -uo pipefail
export HF_HUB_DISABLE_XET=1
DEST="$HOME/MLXLoRAStudio/models"
fetch () {
  local repo="$1" name="$2"; shift 2
  echo "=== [$(date +%H:%M:%S)] START $repo -> $name"
  if hf download "$repo" "$@" >/tmp/hfdl_$$.log 2>&1; then
    local snap
    snap=$(python3 -c "from huggingface_hub import snapshot_download;print(snapshot_download('$repo',local_files_only=True))")
    ln -sfn "$snap" "$DEST/$name"
    echo "=== [$(date +%H:%M:%S)] DONE  $name -> $snap"
  else
    echo "=== [$(date +%H:%M:%S)] FAILED $repo"; tail -5 /tmp/hfdl_$$.log
  fi
}
fetch "DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP" "Qwen3.5-9B-DefiantFable-bf16" --exclude "backup/*"
fetch "mlx-community/gemma-4-12B-it-4bit" "gemma-4-12B-it-4bit"
echo "=== [$(date +%H:%M:%S)] ALL DONE ==="
