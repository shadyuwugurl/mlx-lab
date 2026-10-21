"""Probe: does the classic unsloth save API work on the MLX (Apple Silicon) path?

Run with: .venv-unsloth/bin/python scripts/unsloth_mlx_save_test.py
"""
import os
import traceback

import unsloth

MODEL = os.path.expanduser(
    "~/MLXLoRAStudio/models/Qwen3.5-9B-DefiantFable-4bit"
)
OUT = os.path.expanduser("~/mlx-lab-runs/unsloth-mlx-save")

print("DEVICE_TYPE           :", getattr(unsloth, "DEVICE_TYPE", None))
print("has save_pretrained   :", hasattr(unsloth, "save_pretrained"))
print("has unsloth_save_model:", hasattr(unsloth, "unsloth_save_model"))
print("module of save_pretrained:",
      getattr(getattr(unsloth, "save_pretrained", None), "__module__", None))

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL, max_seq_length=256, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=["q_proj", "v_proj"],
    use_gradient_checkpointing="mlx",
)
print("model type:", type(model).__module__ + "." + type(model).__name__)

for method in ("lora", "merged"):
    out_dir = f"{OUT}-{method}"
    fn = getattr(unsloth, "save_pretrained", None)
    if fn is None:
        print(f"[{method}] no save_pretrained attribute on unsloth")
        continue
    try:
        fn(model, out_dir, save_method=method)
        produced = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []
        print(f"[{method}] OK -> {out_dir} contents={produced}")
    except Exception as exc:  # noqa: BLE001
        print(f"[{method}] FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
