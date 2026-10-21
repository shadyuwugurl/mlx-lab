# Upstream pull request index

Every pull request below was found by running the code on an Apple M5 (32 GB), not by
reading the source. Each has a local reproduction, a fix, and a verification run recorded
in this repository.

Forks live under [`shadyuwugurl`](https://github.com/shadyuwugurl).

## mlx-lm-lora — https://github.com/Goekdeniz-Guelmez/mlx-lm-lora

| PR | Title | What it fixes |
|---|---|---|
| [#106](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora/pull/106) | Return the evaluated token count from `evaluate_sft` | `evaluate_sft()` returned a bare `float`, but `train.py` unpacks `test_loss, test_ntokens = evaluate_sft(...)` → `TypeError: cannot unpack non-iterable float object`. Every SFT run that reached the test split died there. |
| [#107](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora/pull/107) | Add a fallback chat template so base checkpoints stay trainable | `trainer/datasets.py` calls `tokenizer.apply_chat_template(...)` with no fallback, so a base checkpoint without a chat template raises `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set`. `google/gemma-4-12B` is such a checkpoint. Adds `mlx_lm_lora/chat_template.py` + `ensure_chat_template()`, called from `utils.from_pretrained` and `train.load_judge_model`. |

## MLX-LoRA-Studio — https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio

| PR | Title | What it fixes |
|---|---|---|
| [#23](https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio/pull/23) | Normalize Hugging Face dataset/model URLs to bare repo ids | The "Add Hugging Face dataset" field stored `https://huggingface.co/datasets/owner/name` verbatim; `datasets.load_dataset()` then treated it as a local path → `FileNotFoundError: Couldn't find any data file at <cwd>/https:/huggingface.co/...`. |
| [#24](https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio/pull/24) | Add a CI workflow that runs the backend test suite | The repo had no CI at all, so the Python backend test suite never ran on any push. The job is hermetic: the suite stubs `mlx`, `mlx_lm` and `mlx_lm_lora`, so it needs only `numpy`. |
| [#25](https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio/pull/25) | Do not discard a trained adapter when the test pass fails | `Backend/training_runner.py` ran the test-split evaluation outside any `try`, so an evaluation exception aborted `run()` before the adapter export — a fully trained adapter was thrown away. |
| [#26](https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio/pull/26) | Fail fast with an actionable message when MLX is not installed | The default interpreter is `/usr/bin/python3`, which has no `mlx`; the job failed with a bare `ModuleNotFoundError: No module named 'mlx'` and no hint about which interpreter to pick. |

## mlx — https://github.com/ml-explore/mlx

| PR | Title | What it fixes |
|---|---|---|
| [#4732](https://github.com/ml-explore/mlx/pull/4732) | feat(nn): add `freqs` and `partial_rotary_factor` support to `RoPE` | `mx.fast.rope` already accepts a custom `freqs` array, but `mlx.nn.RoPE` did not expose it, so partial/proportional rotary embeddings were impossible to express through the public layer API. `mlx-lm` had to reimplement six RoPE variants in Python (`mlx_lm/models/rope_utils.py`). gemma-4 (`partial_rotary_factor 0.25`) and qwen3.5 both need it. |

> Note: the canonical upstream is `ml-explore/mlx`. There is no `apple/mlx` repository.

## unsloth — https://github.com/unslothai/unsloth

| PR | Title | What it fixes |
|---|---|---|
| [#14457](https://github.com/unslothai/unsloth/pull/14457) | fix(cli): make `unsloth doctor` MLX-aware on Apple Silicon | `unsloth doctor` only knew about CUDA/ROCm. On a healthy Apple Silicon machine it printed `=== GPUs (0) ===`, `PROBLEMS FOUND` and `No GPU is visible to torch. Check nvidia-smi…`, and **exited 1** — even though `unsloth train` works there through the MLX backend. Any CI gate on `unsloth doctor` failed on Macs. |

## Verifying the claims

Every reproduction above is scripted in this repository:

```
scripts/verify_model.py      # load + generate for each target model
scripts/verify_all.sh        # all four models
scripts/run_studio.py        # drive the installed MLX LoRA Studio backend
scripts/run_sft_matrix.sh    # the three Studio SFT runs
scripts/unsloth_mlx_run.py   # unsloth MLX LoRA training on a target model
```

See `docs/MODEL_MATRIX.md` for the per-model measurements and `docs/FINDINGS.md` for the
bugs that are not yet filed upstream.
