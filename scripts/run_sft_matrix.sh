#!/bin/bash
# Studio SFT evidence for the remaining target models.
cd /Users/charlie/Documents/mlx-lab || exit 1
PY="$HOME/MLXLoRAStudio/envs/mlx-lora-studio/bin/python"
[ -x "$PY" ] || PY="$(pwd)/.venv/bin/python"
# The app puts its bundled vendor checkout on PYTHONPATH for child processes.
# The vendor root is .../StudioSupport/vendor and the package dir is
# vendor/mlx_lm_lora (there is no "-lora" suffix on the directory).
VENDOR="/Applications/MLX LoRA Studio.app/Contents/Resources/StudioSupport/vendor"
export PYTHONPATH="$VENDOR${PYTHONPATH:+:$PYTHONPATH}"
run () {
  local name="$1"
  echo "=== [$(date +%H:%M:%S)] SFT $name"
  "$PY" scripts/run_studio.py "$HOME/mlx-lab-runs/$name/run_spec.json" \
    > "$HOME/mlx-lab-runs/$name/run.log" 2>&1
  echo "    exit=$? (log: ~/mlx-lab-runs/$name/run.log)"
}
run "qwen35-defiant-sft"
run "gemma4-it-sft"
echo "=== [$(date +%H:%M:%S)] SFT MATRIX DONE ==="
