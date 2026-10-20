# MODEL_MATRIX.md — the three verification models

Every ported repo must demonstrate itself on at least one of these three models, on the local
Apple M5 (32 GB unified memory, 10 cores, Metal).

All three are **ungated** on the Hub, so the downloads need no token.

| # | Upstream repo | Architecture | `model_type` | bf16 | Local artifact we train on | Size |
|---|---|---|---|---|---|---|
| M1 | `LiquidAI/LFM2.5-24B-A2B` | `Lfm2MoeForCausalLM` | `lfm2_moe` | 47.96 GB | `mlx-community/LFM2.5-24B-A2B-OptiQ-4bit` | 16.69 GB |
| M2 | `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP` | `Qwen3_5ForConditionalGeneration` | `qwen3_5` | 19.36 GB | local conversion of the upstream repo → MLX 4-bit | ≈ 5.5 GB |
| M3 | `google/gemma-4-12B` | `Gemma4UnifiedForConditionalGeneration` | `gemma4_unified` | 23.95 GB | `mlx-community/gemma-4-12B-4bit` | 11.02 GB |

Local paths: `~/MLXLoRAStudio/models/<name>` (symlinks into the HF cache — the cache is never copied).

| local name | points at |
|---|---|
| `LFM2.5-24B-A2B-4bit` | `mlx-community/LFM2.5-24B-A2B-OptiQ-4bit` |
| `gemma-4-12B-4bit` | `mlx-community/gemma-4-12B-4bit` |
| `Qwen3.5-9B-DefiantFable-bf16` | `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP` (converted in place to a `-mlx-4bit` dir) |

## mlx-lm support (already present, no port needed)

`mlx-lm` 0.32.2 dispatches models through `MODEL_REMAPPING` in `mlx_lm/utils.py` and then
`importlib.import_module("mlx_lm.models." + model_type)`:

* `lfm2_moe` → `mlx_lm/models/lfm2_moe.py`
* `qwen3_5` → `mlx_lm/models/qwen3_5.py` (carries `PipelineMixin` → the model is **multimodal**)
* `gemma4_unified` → remapped to `gemma4` → `mlx_lm/models/gemma4.py`
  (comment in upstream source: *"encoder-free multimodal variant; vision/audio weights stripped by sanitize()"*)

So **all three architectures already load and generate under `mlx-lm` 0.32.2**. The missing pieces are
in the Studio layer (arch whitelist, chat templates, dataset handling) — that is where our upstream PRs land.

## Model-specific notes

### M1 — LFM2.5-24B-A2B (MoE)

* `hidden_size 2048`, `num_hidden_layers 40`, `num_experts 64`, `num_experts_per_tok 4`, `vocab_size 128000`.
* The OptiQ-4bit artifact is **mixed-bit**: `group_size 64`, `bits 4` affine for the expert MLPs
  (`switch_mlp.{gate,up,down}_proj`), `bits 8` for the sensitive layers (embeddings, conv projections,
  attention projections, router gates).
* Memory: 16.69 GB of quantized weights on a 32 GB machine. Training must use `--grad-checkpoint`,
  `--batch-size 1`, and a bounded `--max-seq-length`.
* LoRA target recommendation: attention projections only (`q/k/v/o`) — LoRA on 64 experts per layer
  multiplies the adapter count by 64 and blows the memory budget.
* This is the heaviest model and the best stress test for memory behaviour.

### M2 — Qwen3.5-9B Defiant Fable (multimodal, MTP head)

* `hidden 4096`, `num_hidden_layers 32`, `vocab_size 248320`, `num_attention_heads 16`.
* The repo carries a `vision_config` and `image_token_id` / `video_token_id` → it is a **multimodal**
  `Qwen3_5ForConditionalGeneration`, with `preprocessor_config.json` and `video_preprocessor_config.json`.
* It also carries `model-mtp-restored.safetensors` (multi-token-prediction head) alongside the 4-shard
  index, plus a `backup/` directory of the pre-merge originals.
* No 4-bit artifact exists for this exact merge, so we convert it locally with
  `mlx_lm.convert --hf-path <snapshot> -q --q-bits 4 --q-group-size 64 --dtype bfloat16`.
  The MTP head is not part of the SFT loss path and is expected to be dropped by the conversion; the
  conversion output must be diffed against the source index to confirm nothing else is lost.
* This model is the "unsloth-style" merge the user already works with — it is the primary target for
  abliteration and merge ports.

### M3 — gemma-4-12B (encoder-free multimodal)

* `Gemma4UnifiedForConditionalGeneration`, 11.02 GB as 4-bit.
* Needs a non-standard chat template; the Studio has to render it (ties into the `huggingface/swift-jinja`
  port in T1).

## Verified status

| model | downloaded | loads in `mlx_lm` | generates | Studio trains |
|---|---|---|---|---|
| M1 LFM2.5-24B-A2B-4bit | pending | pending | pending | pending |
| M2 Qwen3.5-9B-DefiantFable | pending (convert after download) | pending | pending | pending |
| M3 gemma-4-12B-4bit | pending | pending | pending | pending |

This table is updated in place as each cell is verified; the "verified" rows are the evidence trail for
the goal's completion criteria.
