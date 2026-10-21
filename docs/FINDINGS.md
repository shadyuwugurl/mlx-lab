# Findings

Bugs and behaviours found by running the stack on an Apple M5 (32 GB, macOS 26).
Filed upstream ones live in `docs/PR_INDEX.md`; this file records everything else,
including the findings that are **not** bugs so nobody re-files them.

## 1. MLX float32 matmul is TF32 by default on Apple Silicon

`mx.matmul` / `mx.fast.metal` float32 on this M5 has a relative error of **7.3e-4**
against a float64 numpy reference. The *same operation* on the MLX CPU stream gives
**1.03e-6** — the Metal path is ~700x less accurate than MLX's own CPU path. Every
other float32 op tested (sum, softmax, tanh, mul) is exact to 1e-7.

Root cause, from the MLX source (`mlx/utils.h`):

```cpp
inline bool enable_tf32() {
  static bool enable_tf32_ = get_var("MLX_ENABLE_TF32", 1);  // default ON
  return enable_tf32_;
}
```

It is read once at startup and used by `mlx/backend/metal/matmul.cpp`,
`quantized.cpp` and `scaled_dot_product_attention.cpp`. Setting
`MLX_ENABLE_TF32=0` drops the error to **5.1e-7**.

Consequence: **MLX's own Python test suite fails 1441 tests with the default settings**
and **1 test** with `MLX_ENABLE_TF32=0`:

```
# default
1441 failed, 870 passed        (test_linalg 10, test_fast_sdpa 8, test_quantized 5, ...)

# MLX_ENABLE_TF32=0
1 failed, 910 passed, 15279 subtests passed
```

Upstream's own `python/tests/run.py` sets `os.environ["MLX_ENABLE_TF32"] = "0"`, so the
default that every user gets is not what CI exercises. Upstream issue
[#4595](https://github.com/ml-explore/mlx/issues/4595) asks for a per-call / per-scope
opt-out and notes the current escape hatch "applies to the whole process, is read once
at startup, and makes other fp32 work much slower".

**Not filed as a PR here** — the fix is in C++ and this machine has no Metal compiler,
so it cannot be built or tested locally. `docs/PR_INDEX.md` #4732 mentions it in a note.

### Practical advice

Anything doing weight surgery or numerics on Apple Silicon should either set
`MLX_ENABLE_TF32=0` or force the CPU stream for the precision-sensitive ops. This is
already the invariant recorded in the `spark-pipeline` repository's `MEMORY.md` for
float32 matmuls on this M5.

## 2. `mlx.nn.RoPE` cannot express partial / proportional RoPE

`mx.fast.rope(a, dims, *, traditional, base, scale, offset, freqs=None, stream)` — the
C++ op supports a custom `freqs` array. `mlx.nn.RoPE` did not expose it, so
partial-rotary embeddings were unreachable through the public layer API.

`mlx-lm` works around this in Python (`mlx_lm/models/rope_utils.py`): `SuScaledRoPE`,
`Llama3RoPE`, `YarnRoPE`, `ProportionalRoPE`, `DynamicNTKScalingRoPE`,
`initialize_rope()` and `apply_yarn_mscale()`. `ProportionalRoPE` builds `freqs` with
`mx.inf` padding for the non-rotated dimensions.

This was verified against the HF reference
(`transformers/modeling_rope_utils.py::_compute_proportional_rope_parameters`) — the
mlx-lm implementation is correct; only the MLX layer was missing. Filed as
`docs/PR_INDEX.md` #4732.

## 3. `mlx-lm-lora` needs a chat template to train anything

`trainer/datasets.py` calls `tokenizer.apply_chat_template(...)` at 15 call sites with
no fallback. A base checkpoint without a chat template raises
`ValueError: Cannot use chat template functions because tokenizer.chat_template is not set`.

`google/gemma-4-12B` is such a checkpoint (`google/gemma-4-12B-it` is the instruct
sibling that has one). Filed as `docs/PR_INDEX.md` #107.

## 4. `evaluate_sft` returns a float but its caller unpacks a tuple

`mlx_lm_lora/trainer/sft_trainer.py::evaluate_sft` returned `(all_losses / ntokens).item()`;
`mlx_lm_lora/train.py` does `test_loss, test_ntokens = evaluate_sft(...)` →
`TypeError: cannot unpack non-iterable float object`. Present in v3.1.2, v5.8.6 and
`origin/main`. Filed as `docs/PR_INDEX.md` #106.

Worth noting: this bug does **not** by itself lose the trained adapter —
`train_sft()` saves the final adapter at the end of its own loop, and `train.py` calls
`train_sft()` before `evaluate_sft()`. The adapter loss observed in the Studio came
from the dataset-URL bug (#23 below) and from the Studio's own unguarded evaluation
(#25 below).

## 5. The Studio stores Hugging Face URLs verbatim

`AppStore.addCustomHFAsset` trims the string and, if it contains a `/`, keeps it as-is.
A pasted `https://huggingface.co/datasets/HuggingFaceH4/no_robots` is stored unchanged
and later handed to `datasets.load_dataset()`, which resolves it against the working
directory → `FileNotFoundError: Couldn't find any data file at <cwd>/https:/huggingface.co/...`.

The dataset field expects a **bare repo id** (`owner/name`). Filed as
`docs/PR_INDEX.md` #23.

## 6. The Studio's default interpreter has no MLX

`AppStore` ships `pythonExecutable = "/usr/bin/python3"` (the Xcode Command Line Tools
Python, 3.9.6), which has no `mlx`. A job then fails with a bare
`ModuleNotFoundError: No module named 'mlx'` and no indication of which interpreter to
select. Filed as `docs/PR_INDEX.md` #26.

## 7. `unsloth doctor` is blind on Apple Silicon

`unsloth doctor` only knows CUDA/ROCm. On a healthy M5 with the MLX backend working:

```
=== GPUs (0) ===

=== Verdict: PROBLEMS FOUND ===
- No GPU is visible to torch. Check `nvidia-smi`, CUDA_VISIBLE_DEVICES, and that torch
  was installed with CUDA or ROCm support.
```

Exit code **1** ("Exit status 1 means a problem was found"), so any CI gate on it fails
on Macs. Filed as `docs/PR_INDEX.md` #14457.

## 8. `unsloth`'s MLX backend is already solid (not a bug)

Worth recording, because the surface area suggests otherwise:

* `unsloth.UnslothTrainer` is exported on the MLX path (`unsloth/__init__.py`, inside the
  `if _IS_MLX:` branch).
* `unsloth_zoo` ships a **triton stub** and injects it, so `import unsloth.kernels`
  succeeds on macOS even though `triton` is not installed.
* The 306 `torch.cuda` call sites and the 11 Triton kernel files under `unsloth/kernels/`
  are on the **CUDA path only** — the MLX path never imports torch.
* The save API exists: the zoo binds `model.save_pretrained` and
  `model.save_pretrained_merged` onto MLX models.
* `unsloth/dataprep` imports cleanly on macOS.

An end-to-end unsloth MLX LoRA run on the exact DavidAU target model succeeded on the
M5 (loss 1.4718 → 2.7413, 24.4 tok/s, peak 5.85 GB, `compile` reported
`supported_verified`).

## 9. The `hf` CLI's xet transport can hang at 0 KB/s

Downloading the DavidAU repo with `hf download` stalled with the blob frozen at
67,028,294 bytes for over four minutes, the xet cache not growing, and 99 established
sockets. The same download with `HF_HUB_DISABLE_XET=1` (plain HTTP) sustained
**13.9 MB/s**. Two other repos downloaded fine over xet, so this is repo-specific.

Use `HF_HUB_DISABLE_XET=1` when a large `hf download` makes no progress.

## 10. The released Studio app ships an older backend than upstream pins

The installed app is v2.0.3 (published 2026-08-26) and bundles **mlx-lm-lora 3.1.2**.
`mlx-lm-lora` 5.8.6 was only published on 2026-10-05 — after the app release.
Upstream `main` already pins 5.8.6 in `Backend/requirements.txt` and
`script/build_and_run.sh`, and `.github/workflows/release.yml` asserts that the bundled
`_version.py` contains `__version__ = "5.8.6"`.

**This is not an upstream bug** — it is release cadence. The delta between the two
versions is +3775 / -1366 lines over 21 files, adding `klpo_trainer`, `dsla_trainer`,
`fast_vjp`, `long_context` and `online_rl_utils`, and reworking the grpo/ppo/dpo/xpo/rlhf
trainers.

Local repair used here: replace the bundled vendor tree with `git archive v5.8.6` and
re-sign the bundle ad-hoc (`codesign -f -s - --deep --force`). The app puts the vendor
root on `PYTHONPATH` for its child processes, so the bundle's copy shadows any
site-packages copy.
