## Problem

Base (non-instruct) checkpoints are frequently used for LoRA fine-tuning, but they
ship **without a chat template**. Every dataset builder in
`mlx_lm_lora/trainer/datasets.py` calls `tokenizer.apply_chat_template(...)`
without a fallback, so training such a model aborts before the first step:

```
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed! For information about writing templates and setting the tokenizer.chat_template attribute, please see the documentation at https://huggingface.co/docs/transformers/main/en/chat_templating
```

### Reproduction

```bash
# any base checkpoint without a chat template, e.g. google/gemma-4-12B
python -m mlx_lm_lora.train \
  --model google/gemma-4-12B \
  --data HuggingFaceH4/no_robots \
  --iters 2
```

The traceback (abridged):

```
  File ".../mlx_lm_lora/trainer/datasets.py", line 36, in __init__
    prompt_tokens = tokenizer.apply_chat_template(
  File ".../mlx_lm_lora/trainer/datasets.py", line 749, in load_hf_dataset
    dataset = load_dataset(data_id)
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set ...
```

## Fix

Add `mlx_lm_lora/chat_template.py` with:

* `DEFAULT_CHAT_TEMPLATE` — a minimal, tokenizer-agnostic Jinja template that
  renders plain-text roles (`user: ...\nassistant: `). It deliberately avoids
  special tokens such as `<|im_start|>` because they are not present in every
  vocabulary.
* `has_chat_template(tokenizer)` — works with both raw Hugging Face tokenizers
  and the `mlx_lm` `TokenizerWrapper`.
* `ensure_chat_template(tokenizer)` — installs the fallback **only** when the
  tokenizer has no template, and keeps the `TokenizerWrapper`'s cached
  `has_chat_template` flag consistent.

The helper is called at the two places where a tokenizer is loaded:

* `mlx_lm_lora/utils.py::from_pretrained` (the training model)
* `mlx_lm_lora/train.py::load_judge_model` (the judge model)

A checkpoint that already provides a template is never modified.

## Verification

End-to-end on an Apple M5 (32 GB) with the base `google/gemma-4-12B`
(MLX 4-bit, `mlx-community/gemma-4-12B-4bit`), which previously failed with the
`ValueError` above:

```
[Studio] Text SFT LORA run | .../gemma-4-12B-4bit | batch 1 | lr 1e-05 | optimizer lion
Iter 1 | Train | loss 3.485 | 21 tok/s | 11.27 GB
Iter 2 | Train | loss 2.301 | 189 tok/s | 11.38 GB
Test Results:
  Loss:       3.053
  Perplexity: 21.175
  Tokens:     102
[Studio] Run complete. Outputs saved under .../adapters
```

`adapters.safetensors` is written (10.9 MB) and `adapter_config.json` is created.

## Tests

`tests/test_chat_template.py` (stdlib `unittest`, matching the existing suite):

* the default template renders roles and the generation prompt
* `has_chat_template` detects a missing template
* `ensure_chat_template` installs a template when missing
* `ensure_chat_template` leaves an existing template untouched
* `ensure_chat_template` patches a wrapped tokenizer and refreshes the cached flag
* `ensure_chat_template` is a no-op when `template=None`

Full suite: `python -m unittest discover -s tests` → **155 tests, OK**.
