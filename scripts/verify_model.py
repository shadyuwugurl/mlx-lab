"""Load an MLX model and record arch, quant, size and a seed generation."""
import json, sys, time, pathlib
import mlx.core as mx
from mlx_lm import load, generate

path = sys.argv[1]
prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
mx.random.seed(0)
t0 = time.time()
model, tokenizer = load(path)
load_s = time.time() - t0
cfg = model_config = getattr(model, "args", None)
info = {
    "path": path,
    "load_seconds": round(load_s, 2),
    "model_type": getattr(cfg, "model_type", None),
    "quantization": getattr(cfg, "quantization", None),
    "hidden_size": getattr(cfg, "hidden_size", None),
    "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
    "vocab_size": getattr(cfg, "vocab_size", None),
}
try:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    info["template"] = "chat"
except Exception as e:
    text = prompt
    info["template"] = f"raw ({type(e).__name__})"
t0 = time.time()
out = generate(model, tokenizer, prompt=text, max_tokens=48, verbose=False)
info["generate_seconds"] = round(time.time() - t0, 2)
info["prompt"] = prompt
info["completion"] = out if isinstance(out, str) else out.text
size = sum(p.stat().st_size for p in pathlib.Path(path).rglob("*") if p.is_file())
info["on_disk_bytes"] = size
print(json.dumps(info, indent=2, default=str))
