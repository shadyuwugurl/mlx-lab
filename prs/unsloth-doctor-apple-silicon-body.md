## What

`unsloth doctor` only knew how to probe CUDA / ROCm GPUs. On an Apple Silicon Mac — where Unsloth trains through the MLX backend (`unsloth_zoo.mlx`) — it reported a **false failure**:

```
$ unsloth doctor
=== Versions ===
torch 2.14.1
CUDA (torch build) None
driver unknown
NCCL unavailable

=== GPUs (0) ===

=== Verdict: PROBLEMS FOUND ===
- No GPU is visible to torch. Check `nvidia-smi`, CUDA_VISIBLE_DEVICES, and that torch was installed with CUDA or ROCm support.
$ echo $?
1
```

The CLI documents *"Preflight checks for this machine. Exit status 1 means a problem was found"*, so on a perfectly healthy Mac `unsloth doctor` exits **1** and points at `nvidia-smi`. Any CI step that gates on `unsloth doctor` fails on Apple Silicon even though `unsloth train` works there.

## Why this is a real gap, not a cosmetic one

The MLX backend is a first-class path in this repo — `unsloth train`, `unsloth chat`, `unsloth inference` and `unsloth export` all detect it and run on Apple Silicon. `doctor` is the one command that was never taught about it, and it is the command users reach for first when something goes wrong.

This is the same machine, same install, same process — only the doctor was blind:

```
$ unsloth train --model <mlx-4bit-model> ...
Step 1/2 Loss 1.4718 (7 tok/s, peak 5.39 GB)
Step 2/2 Loss 2.7413 (120 tok/s, peak 5.85 GB)
...
compile_support_state='supported_verified'
```

## The fix

`doctor` now asks the repository's own gate which backend is live (`studio.backend.core.training.training.should_use_mlx_training_backend`) and, when MLX is active, runs an MLX preflight instead of the CUDA/NCCL one:

```
$ unsloth doctor
=== Versions ===
platform Darwin arm64
backend mlx
mlx 0.32.4
unsloth_zoo 2026.10.7
Metal available

=== Device ===
  Apple M5  applegpu_g17g
  memory 32.0 GiB
  recommended working set 24.96 GiB

=== Verdict: OK ===
- Apple Silicon GPU through the MLX backend. Multi-GPU, NCCL and peer-copy checks do not apply to this backend, so they were skipped.
$ echo $?
0
```

Details:

- **`mlx_backend_active()`** — honours `UNSLOTH_FORCE_GPU_PATH=1`, requires an importable `mlx`, and defers to the repo's existing `should_use_mlx_training_backend()` gate. Returns `False` on any error, so the CUDA path stays the default everywhere else.
- **`collect_mlx_info()`** — collects the platform, the `mlx` version (via `importlib.metadata`, since `mlx` exposes no `__version__`), `unsloth_zoo`'s version, whether Metal is available, and the device name / architecture / memory from `mx.device_info()`. It never imports `torch` and never raises; failures land in an `errors` list that is printed as notes.
- **`verdict_mlx(info)`** — a **pure** function, in the same style as the existing `verdict()` (*"Pure, so every branch is testable without GPUs"*). It reports healthy only when `mlx`, `unsloth_zoo.mlx` and Metal are all present, and otherwise returns one actionable hint (how to install `mlx`, how to upgrade `unsloth_zoo`, or that Metal is unavailable).
- **The CUDA path is byte-identical.** `_gpu_checks()` is untouched; the only change is the branch that chooses between `_mlx_checks()` and `_gpu_checks(timeout)`.

## Tests

New `unsloth_cli/tests/test_doctor_mlx.py` — 12 tests covering the healthy Apple Silicon case, each failure mode (missing `mlx`, too-old `unsloth_zoo`, no Metal, no device), the `UNSLOTH_FORCE_GPU_PATH` escape hatch, `collect_mlx_info()` never raising, and that `doctor` takes the MLX path and exits 0 while still taking the CUDA path when MLX is inactive.

The two existing CUDA tests in `test_doctor_gpus.py` are pinned to the CUDA path with a one-line monkeypatch, because they otherwise would pick up the MLX path when the suite runs on an Apple Silicon host.

```
$ pytest unsloth_cli/tests/test_doctor_mlx.py unsloth_cli/tests/test_doctor_gpus.py -q
46 passed in 2.48s
```

All of the new tests run without a GPU.

## Notes

- The MLX path is not new — `unsloth_zoo.mlx` has shipped `MLXTrainer` / `FastMLXModel` for a while, and this repo already routes `train`, `chat`, `inference` and `export` to it. This PR only teaches the preflight about it.
- `mx.metal.device_info()` is deprecated in favour of `mx.device_info()`; the code prefers the new API and keeps the old one only as a fallback.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
