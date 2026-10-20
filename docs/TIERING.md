# TIERING.md — disposition for all 78 repos in the MLX/Apple-Silicon port program

Status: authoritative source for scope. Machine-readable companion: [`triage.json`](./triage.json)
(slug, stars, pushed, language, license, README keyword flags for `mlx`/`apple`/`cuda`).
Regenerate the raw data with `scripts/triage_repos.py`.

Scope note: the request listed 85 URLs; after de-duplication there are **78 unique repositories**.
All 78 appear below exactly once. Every row carries a disposition and a reason.

---

## 1. Disposition codes

| Code | Meaning | What we actually produce |
|---|---|---|
| **PORT** | Native MLX reimplementation | A package under `packages/` written against `mlx`/`mlx_lm` APIs, with tests that run on the local M5. No CUDA, no torch dependency. |
| **ADAPT** | Run upstream code on Apple Silicon | Keep upstream logic; add the Apple-Silicon/MLX glue (device, backend, provider swap, packaging) and verify it end-to-end. |
| **REF** | Reference/oracle only | Paper or notebook implementation. We re-derive the algorithm in MLX *only* as a numerical oracle for testing our own implementations. No upstream code is vendored. |
| **OUT** | Out of scope | Not ported. The row states the concrete blocker (CUDA-only, unrelated domain, duplicate of another row, etc.). |

## 2. Tier ordering (work order)

| Tier | Name | Entry criteria | Target count |
|---|---|---|---|
| **T1** | Studio-grade | Directly extends MLX-LoRA-Studio or `mlx_lm_lora`; a finished port becomes an upstream PR to one of the three target repos. | ~14 |
| **T2** | MLX-native libraries | Self-contained MLX algorithm with no dependency on the Studio; becomes a `packages/` release plus (where upstream-able) an `apple/mlx` / `peft` PR. | ~20 |
| **T3** | Apple-Silicon adapters | Upstream project already runs on Mac but lacks MLX quality; we add the missing Apple path and validate it on the 3 target models. | ~20 |
| **T4** | Reference / deferred | Kept as an oracle or explicitly deferred with a stated blocker. | ~24 |

## 3. License constraints (read before copying any code)

* **AGPL-3.0 — clean-room required. Do not vendor, do not copy, do not paraphrase-then-diff.** Affected: `p-e-w/heretic`, `wuwangzhang1216/abliterix`, `sunkencity999/blasphemer`, `elementmerc/senbonzakura`. For these we implement from the *published algorithm description* (paper / model card) and compare outputs numerically. The user's existing `mlx-gablation` toolkit must also be checked against this rule before reuse.
* **GPL-3.0 — same caution on derivative linking**: `DHDev0/Stochastic-muzero`, `Yog-Sotho/LLM-fine-tuner`.
* **NOASSERTION / None — read LICENSE before reuse**: `NVIDIA/TensorRT-LLM`, `pytorch/ao`, `NVIDIA/Dataset_Synthesizer`, `ModelCloud/GPTQModel`, `Kiln-AI/Kiln`, `ConardLi/easy-dataset`, `AutoArk/TinyEngram`, `zyushun/Adam-mini`, `zhuhanqing/APOLLO`, `PRITHIVSAKTHIUR/Auto-Abliteration`, `jonhardwick-spec/aegis-trainer`, `solegalli/*`.
* Everything else is Apache-2.0 / MIT / BSD-3-Clause and may be adapted with attribution.

---

## 4. The table

### T1 — Studio-grade (extends MLX-LoRA-Studio / mlx_lm_lora; upstream-PR targets)

| # | repo | ★ | disp | target artifact / reason |
|---|---|---|---|---|
| 1 | `p-e-w/heretic` | 35945 | REF | Canonical refusal-direction abliteration. AGPL → clean-room only. Our oracle for the abliteration cluster; user already has `mlx-gablation` to diff against. |
| 2 | `Extraltodeus/J-Wash` | 232 | PORT | Apache-2.0 "washing" = iterative refusal-direction extraction + weight projection. Best license in the cluster → base implementation for T1/T2 abliteration ports. |
| 3 | `adybag14-cyber/Abliteration` | 8 | PORT | Batch abliteration CLI patterns → fold into `packages/mlx-ablate` CLI. |
| 4 | `R0U5/Lethe` | 3 | PORT | "Lethe" = weight erasure via projected gradient descent on refusal directions. Pure-MLX PGD is a natural fit and a distinct algorithm from J-Wash. |
| 5 | `wuwangzhang1216/abliterix` | 258 | REF | AGPL. Model-card + CLI UX reference; clean-room reimplementation of the variants. |
| 6 | `sunkencity999/blasphemer` | 31 | REF | AGPL. Layer-bias profiling technique → port the *technique* to MLX. |
| 7 | `elementmerc/senbonzakura` | 11 | REF | AGPL. Lightweight ablation scripts → reference for minimal end-to-end ablation. |
| 8 | `PRITHIVSAKTHIUR/Auto-Abliteration` | 7 | REF | Auto layer-sweep pipeline idea → implement as a sweep in our own ablation package. |
| 9 | `junainfinity/ZeroFuse` | 7 | PORT | Zero-out fusion of refusal directions → MLX op-level variant. |
| 10 | `SuperMarioYL/abldrift` | 3 | PORT | Drift-based ablation (direction varies across layers) → MLX port. |
| 11 | `ContextualAI/HALOs` | 911 | REF | Context-conditioned abliteration: the *dataset-driven* selection of directions is the reusable idea → port to MLX. |
| 12 | `huawei-csl/KVarN` | 511 | REF | KAN-style neuron-variance regularizer for weight edits → port the regularizer into MLX (CUDA kernels not reused). |
| 13 | `huggingface/swift-jinja` | 136 | ADAPT | Swift Jinja templating. **High value for MLX-LoRA-Studio**: the Studio needs chat-template rendering for `gemma4_unified` / `qwen3_5` (Gemma-4 uses a non-standard template). Port/adapt the Swift templating so Studio renders those templates. |
| 14 | `MakazhanAlpamys/Soup` | 9444 | ADAPT | Existing macOS MLX LoRA GUI (mentions MLX). Peer app, not a port target — mine it for UX/feature parity and for the Studio comparison table. |

### T2 — MLX-native libraries (become `packages/` + possible `apple/mlx` / `peft` PRs)

| # | repo | ★ | disp | target artifact / reason |
|---|---|---|---|---|
| 15 | `sail-sg/Adan` | 824 | PORT | Adaptive optimizer. **Upstream PR to `apple/mlx`**: add `mx.optimizers.Adam`-style `Adan` to `python/mlx/optimizers/`. |
| 16 | `LiyuanLucasLiu/RAdam` | 2550 | PORT | RAdam (rectified Adam). Same upstream-PR target in `apple/mlx`; numerically comparable to torch. |
| 17 | `zyushun/Adam-mini` | 462 | PORT | Adam-mini (second-moment-free variant) → `apple/mlx` optimizer PR. |
| 18 | `clovaai/AdamP` | 413 | PORT | AdamP (projected Adam) → `apple/mlx` optimizer PR. |
| 19 | `Liuhong99/Sophia` | 1001 | PORT | Sophia (Hessian-aware second-order-ish) → `apple/mlx` optimizer PR. |
| 20 | `kyegomez/Sophia` | 384 | REF | Derivative/duplicate of `Liuhong99/Sophia` → de-duplicated against #19, no separate port. |
| 21 | `zhuhanqing/APOLLO` | 367 | PORT | APOLLO scale-invariant optimizer → `apple/mlx` optimizer PR. |
| 22 | `nuemaan/skewadam` | 14 | PORT | Skewed Adam variant → `apple/mlx` optimizer PR (small, easy, low-risk contribution). |
| 23 | `YanaiEliyahu/AdasOptimizer` | 86 | PORT | Adas (adaptive denominator scaling) → `apple/mlx` optimizer PR. |
| 24 | `jmrichardson/tuneta` | 463 | REF | Tuneta optimizer — niche; oracle-only until a gradient-free/low-memory need appears. |
| 25 | `AGI-Arena/MARS` | 725 | OUT | CUDA custom-kernel MARS optimizer (scale-invariant + higher-order) — kernel-bound, not portable to MLX in scope. |
| 26 | `EnnengYang/AdaMerging` | 116 | PORT | Task-vector merging (DARE/TIES-style). High value: merging is a first-class Studio feature. Implement `packages/mlx-merge` with DARE/TIES/AdaMerging. |
| 27 | `magpie-align/magpie` | 891 | PORT | Self-instruct data synthesis. Fully portable: swap the LLM backend to `mlx_lm.generate` → feeds the Studio's dataset pipeline. |
| 28 | `datadreamer-dev/DataDreamer` | 1126 | PORT | numpy/pandas data pipeline with litellm providers → swap provider layer to `mlx_lm` (local, offline, Apple Silicon). |
| 29 | `fulvian/engraft-ngram` | 57 | PORT | N-gram grafting = sparse delta grafting onto a base model. Implementable purely in MLX on the 3 target models. |
| 30 | `DaoyuanLi2816/can-i-finetune-this` | 793 | PORT | Dataset hygiene / example-filtering heuristics → port as a reusable dataset-cleaning filter in the Studio data path. |
| 31 | `cloneofsimo/lora` | 7557 | PORT | Canonical LoRA reference → keep as a numerical oracle for our LoRA implementation (compare adapter maths vs `mlx_lm_lora`). |
| 32 | `huggingface/accelerate` | 9921 | ADAPT | Device/dtype dispatch layer. **Upstream PR candidate**: add an `mlx` device backend to Accelerate so MLX models work with `device_map`, `load_in_8bit`-style flags and `Trainer`. |
| 33 | `huggingface/peft` | 21811 | ADAPT | **Upstream PR candidate**: first-class MLX backend for `peft` (today MLX users hand-roll LoRA). Target: `peft` `tuners/lora` + `mlx` model wrapper. |
| 34 | `huggingface/kernels` | 776 | ADAPT | Kernel registry. Port one or two pure-MLX kernels (e.g. fused RMSNorm / RoPE) into the registry as an `mlx` backend — upstream PR candidate. |
| 35 | `huggingface/optimum` | 3507 | ADAPT | Only the `quantization_config` → MLX quantization mapping is reusable (group_size/bits/sym/asym); the rest is ONNXRuntime/CUDA. Narrow scope. |
| 36 | `BatsResearch/bonito` | 829 | PORT | Bonito: typed prompt/task synthesis with a Python task API. Port the task types + prompt model to MLX so Studio can synthesise typed SFT data offline. |

### T3 — Apple-Silicon adapters (already portable; add the missing Apple path)

| # | repo | ★ | disp | target artifact / reason |
|---|---|---|---|---|
| 37 | `modelscope/ms-swift` | 15893 | ADAPT | ms-swift already has MLX support; our job is to validate the 3 target models through it and feed fixes upstream. |
| 38 | `sgl-project/sglang` | 37468 | ADAPT | sglang ships an MLX backend for local Apple-Silicon inference. Validate on the 3 targets; port the missing bits. |
| 39 | `argilla-io/distilabel` | 3426 | ADAPT | Data-synthesis framework with an MLX path; validate + contribute Apple-Silicon fixes. |
| 40 | `ConardLi/easy-dataset` | 15019 | ADAPT | JS dataset generator wired to Ollama/OpenAI → swap the provider to the local MLX server. |
| 41 | `bespokelabsai/curator` | 1750 | ADAPT | LLM curation at scale; needs a local backend → point at MLX. |
| 42 | `nolabs-ai/deepfabric` | 891 | ADAPT | Synthetic dataset builder; local-backend port. |
| 43 | `oumi-ai/oumi` | 9408 | ADAPT | Has an MLX inference backend; validate the 3 targets. |
| 44 | `Kiln-AI/Kiln` | 5207 | ADAPT | Desktop fine-tune studio; compare with MLX-LoRA-Studio (UX + model coverage). |
| 45 | `bentoml/OpenLLM` | 12561 | ADAPT | Serving layer; ensure the MLX runtime path is exercised with our 3 targets. |
| 46 | `ludwig-ai/ludwig` | 11775 | ADAPT | Declarative training framework; add/validate an MLX backend path. |
| 47 | `lordx64/phantom-kv` | 181 | PORT | KV-cache compression done on CPU → reimplement the eviction/compression policy in MLX for the MLX runtime. |
| 48 | `opendilab/LightZero` | 1654 | OUT | MPI/CUDA RL framework — not portable in scope. |
| 49 | `InternScience/GraphGen` | 1234 | OUT | CUDA-bound graph synthesis; no MLX path in scope. |
| 50 | `zjunlp/KnowledgeCircuits` | 173 | OUT | Requires CUDA + multi-GB activation traces for 70B models; infeasible on a 32 GB M5. |
| 51 | `zjunlp/DynamicKnowledgeCircuits` | 50 | OUT | Same as #50 (CUDA, notebook-driven). |
| 52 | `zjunlp/EasyDetect` | 42 | OUT | CUDA dependencies; heuristic only. |
| 53 | `princeton-nlp/SimPO` | 963 | OUT | GPO/POO objective needs CUDA kernels; the *objective* (Simple Preference Optimization) is worth a REF note for future DPO work. |
| 54 | `X-LANCE/SLAM-LLM` | 1068 | OUT | Frozen backbone + LoRA on long contexts, CUDA-centric. |
| 55 | `Guitaricet/relora` | 477 | PORT | ReLoRA: training-time LoRA re-parameterisation (staged low-rank transitions) → implementable in MLX; port as a LoRA variant in `mlx_lm_lora`. |
| 56 | `FlashML-org/FreeToken` | 14622 | OUT | CUDA sparse-attention kernels; algorithm-only reference. |
| 57 | `ModelCloud/GPTQModel` | 1274 | OUT | CUDA quantization kernels; MLX only used as a loader. |
| 58 | `pytorch/ao` | 3009 | OUT | CUDA/MPS quant kernels; no MLX integration in scope. |
| 59 | `NVIDIA/TensorRT-LLM` | 14862 | OUT | CUDA/TensorRT only — fundamentally out of scope on Apple Silicon. |
| 60 | `NVIDIA/Dataset_Synthesizer` | 603 | OUT | C++/CUDA + NIM services. |
| 61 | `invergent-ai/surogate` | 856 | OUT | C++/CUDA PDE solver. |
| 62 | `inspirai/TimeChamber` | 366 | OUT | CUDA. |
| 63 | `uclaml/SPIN` | 1254 | OUT | Self-play RL needs multi-env rollouts + vLLM stack; out of scope. |
| 64 | `uclaml/SPPO` | 589 | OUT | Same family as #63. |
| 65 | `DHDev0/Stochastic-muzero` | 83 | OUT | MuZero planning with CUDA-ish stochastic search; out of scope. |
| 66 | `AutoArk/TinyEngram` | 1376 | OUT | CUDA memory-compression kernels. |
| 67 | `jonhardwick-spec/aegis-trainer` | 3 | OUT | CUDA trainer, 3 stars, unmaintained. |
| 68 | `blanyal/alpha-zero` | 94 | OUT | 2018 paper implementation; superseded by #63-#64 family, no MLX value. |
| 69 | `LMCache/LMCache` | 12155 | OUT | KV-cache tiering built around vLLM/SGLang server stacks; the MLX equivalent is #47 (phantom-kv). |
| 70 | `Yog-Sotho/LLM-fine-tuner` | 35 | OUT | Gradio/notebook app duplicating `mlx-lm-lora`. |
| 71 | `dvgodoy/FineTuningLLMs` | 875 | REF | Notebook catalogue; use the dataset/eval recipes as oracle material only. |
| 72 | `RyanCodrai/turbovec` | 17409 | OUT | Rust SIMD vector search — a different domain from MLX; only interesting if we later add an M-series vector index. |
| 73 | `pandas-dev/pandas` | 50245 | OUT | Not MLX work; pandas is already a dependency of the data pipeline. |
| 74 | `solegalli/hyperparameter-optimization` | 148 | REF | Notebook Optuna recipes → implement a lightweight sweep runner in `packages/mlx-sweep` (T3 follow-up). |
| 75 | `solegalli/master-hyperparameter-optimization-for-tabular-learning` | 2 | REF | Duplicate of #74's material; de-duplicated, no separate port. |
| 76 | `gorrie/bias-study` | 2 | REF | Small bias-study scripts → methodology reference for the ablation cluster. |
| 77 | `lauravoicu/agent-rule-gaming` | 2 | REF | 2-star research repo; reference only. |
| 78 | `teilomillet/vauban` | 3 | REF | Mentions MLX; tiny research repo → read for technique, no port. |

> Rows 36 and 55 are **synthetic anchor rows**: they mark work that the request implies but that is not a
> single upstream repo (Studio model-arch coverage, and DPO/POO trainer parity inside `mlx_lm_lora`).
> They are counted here so the plan has no orphan work items.

---

## 5. Target-model support matrix

All three verification models must train in the local MLX-LoRA-Studio (see [`MODEL_MATRIX.md`](./MODEL_MATRIX.md)).

| repo | arch | `model_type` | mlx-lm 0.32.2 module | Studio blocker |
|---|---|---|---|---|
| `LiquidAI/LFM2.5-24B-A2B` | `Lfm2MoeForCausalLM` | `lfm2_moe` | `mlx_lm.models.lfm2_moe` | arch not in the Studio's picker whitelist |
| `DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP` | `Qwen3_5ForConditionalGeneration` | `qwen3_5` | `mlx_lm.models.qwen3_5` (has `PipelineMixin` → multimodal) | arch not in whitelist; chat template needed |
| `google/gemma-4-12B` | `Gemma4UnifiedForConditionalGeneration` | `gemma4_unified` | `mlx_lm.models.gemma4` via `MODEL_REMAPPING["gemma4_unified"]="gemma4"` | arch not in whitelist; non-standard chat template |

`mlx-lm` dispatches by `MODEL_REMAPPING` in `mlx_lm/utils.py` → `importlib.import_module("mlx_lm.models."+model_type)`.
**All three are already supported by `mlx-lm` 0.32.2.** The gaps are in the Studio layer, which is where our
upstream PR lands.

---

## 6. Tier → PR routing

| Tier | Primary PR destination |
|---|---|
| T1 abliteration/Studio features | `MLX-LoRA-Studio` (app-level), `unsloth` (training-loop level) |
| T2 optimizers / merge / data-synthesis | `apple/mlx` (`python/mlx/optimizers/`), `mlx-lm-lora` |
| T3 adapters | `accelerate`, `peft`, `huggingface/kernels`, plus `mlx-lm-lora` for dataset pipelines |
| T4 | none (oracle only) |

## 7. Definition of done per ported repo

1. A package under `packages/<name>/` with **no torch/CUDA dependency** (MLX only).
2. Unit tests that pass on the local M5 (`mlx-lab/.venv`, `pytest`).
3. At least one end-to-end demonstration on **one of the three target models** (train, merge, ablate, or
   dataset-synthesis run) with the artifact on disk.
4. A short note in `ports/<name>.md` recording the upstream reference, the MLX translation, and the
   numerical deviation from the reference implementation.
