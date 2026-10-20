# mlx-lab

Native **MLX / Apple Silicon** implementations, and upstream contributions for the tools this
machine actually runs on: [`apple/mlx`](https://github.com/ml-explore/mlx),
[`MLX-LoRA-Studio`](https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio) and
[`unsloth`](https://github.com/unslothai/unsloth).

Everything here is **MLX-only**: no torch, no CUDA, no PyPI wheels that pull a GPU runtime.
Ports are native re-implementations, not thin wrappers.

## Layout

```
packages/     native MLX libraries (abliteration, merge, optimizers, data synthesis, ...)
ports/        one note per ported repo: upstream ref, MLX translation, numerical deviation
prs/          the upstream PR branches + the exact repro for each
docs/         TIERING.md (scope of all 78 repos), MODEL_MATRIX.md (the 3 verification models),
              triage.json (raw repo metadata)
scripts/      model fetch, triage regeneration, verification harnesses
upstreams/    gitignored clones of the upstream repos (read-only references)
models/       gitignored; download logs
.venv/        gitignored shared environment (uv, Python 3.13)
```

## Environment

```bash
uv venv --seed --python 3.13 .venv
uv pip install --python .venv -U mlx mlx-lm mlx-lm-lora matplotlib pytest huggingface_hub
```

Current pins: `mlx 0.32.4`, `mlx-lm 0.32.2`, `mlx-lm-lora 5.8.6`.

Verify the Metal device is live before anything else:

```bash
.venv/bin/python -c "import mlx.core as mx; print(mx.default_device(), mx.metal.is_available())"
# expected: Device(gpu, 0) True
```

## The three verification models

See [`docs/MODEL_MATRIX.md`](docs/MODEL_MATRIX.md). Every port must prove itself on at least one:

| local name | upstream | size |
|---|---|---|
| `~/MLXLoRAStudio/models/LFM2.5-24B-A2B-4bit` | `mlx-community/LFM2.5-24B-A2B-OptiQ-4bit` | 16.69 GB |
| `~/MLXLoRAStudio/models/gemma-4-12B-4bit` | `mlx-community/gemma-4-12B-4bit` | 11.02 GB |
| `~/MLXLoRAStudio/models/Qwen3.5-9B-DefiantFable-mlx-4bit` | `DavidAU/Qwen3.5-9B-…-NEO-IMATRIX-MAX-MTP` (converted) | ≈ 5.5 GB |

```bash
scripts/fetch_models.sh          # downloads + symlinks into ~/MLXLoRAStudio/models
```

## Scope

78 unique repositories from the original list. **All 78 have a disposition and a reason** in
[`docs/TIERING.md`](docs/TIERING.md) — nothing is silently dropped.

| Tier | Meaning | Count |
|---|---|---|
| T1 | Studio-grade: extends MLX-LoRA-Studio or `mlx_lm_lora`, ships as an upstream PR | 14 |
| T2 | MLX-native libraries → `packages/` + `apple/mlx` optimizer/utility PRs | 22 |
| T3 | Apple-Silicon adapters for upstream projects that already run on Mac | 19 |
| T4 | Reference-only or explicitly deferred with a stated blocker | 23 |

### Licence rule

`p-e-w/heretic`, `wuwangzhang1216/abliterix`, `sunkencity999/blasphemer` and
`elementmerc/senbonzakura` are **AGPL-3.0**. Nothing from them is vendored, copied, or
diffed against. They are re-implemented clean-room from the published algorithm and used only as
numerical oracles. `DHDev0/Stochastic-muzero` and `Yog-Sotho/LLM-fine-tuner` are GPL-3.0 and treated
the same way.

## Ground rules

1. No torch, no CUDA, in any package under `packages/`.
2. Every port ships tests that pass on this machine (`pytest`, M5).
3. Every port proves itself end-to-end on at least one of the three models, with the artifact on disk.
4. Public GitHub state is only mutated once a publishable artifact exists — forks live under
   `shadyuwugurl`, and PRs are opened against the real upstreams.
