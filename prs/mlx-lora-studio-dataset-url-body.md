## Problem

The "Add Hugging Face dataset" field accepts anything that contains a `/`, and the value is stored **verbatim** in `customDatasetPaths`:

```swift
// Sources/MLXLoRAStudio/Stores/AppStore.swift
let normalised = trimmed.contains("/")
    ? trimmed
    : trimmed.replacingOccurrences(of: "--", with: "/")
```

So a pasted dataset URL such as

```
https://huggingface.co/datasets/HuggingFaceH4/no_robots
```

is stored as-is. `datasets.load_dataset()` (reached through `mlx_lm_lora`'s `load_hf_dataset`, used for both `args.data` and `args.hf_dataset`) expects a **repo id**; handed a URL it treats the string as a local path and the run dies before the first step:

```
FileNotFoundError: Couldn't find any data file at
/Users/<you>/https:/huggingface.co/datasets/HuggingFaceH4/no_robots.
```

The error is especially confusing because the path shows up as `https:/…` (a single slash) inside the current working directory.

The same happens for a model URL, and it also affects run specs that were written before this fix — the stored preference keeps the URL, so the failure is permanent until the user edits the chip by hand.

## Fix

Two layers, so that both newly-typed values and already-stored ones work:

1. **Swift (`AppStore.addCustomHFAsset`)** — a new `normalizedHFReference(_:)` helper turns a pasted reference into a bare `owner/name` repo id. It handles `https://huggingface.co/datasets/owner/name`, `https://huggingface.co/owner/name`, the `www.` variants, `http://`, trailing slashes, `?`/`#` query fragments, and the existing `owner--name` shorthand. Local filesystem paths and bare repo ids are left untouched.

2. **Python (`Backend/training_runner.py`)** — a new `_normalize_hf_dataset_reference()` helper does the same normalisation, and is applied to `args.data` (before `_prepare_local_dataset_for_trainer`) and to `args.hf_dataset`. This is what makes **already-stored URLs** work again, and it also covers run specs written by hand or by other tooling.

Anything that does not look like an `owner/name` repo id after stripping the URL prefix is passed through untouched, so local paths and existing behaviour are unaffected.

## Tests

`tests/test_training_runner.py` gains two cases:

- `test_hf_dataset_urls_are_normalized_to_repo_ids` — URL, URL with a trailing slash, scheme-less URL, bare repo id, local path.
- `test_normalize_spec_resolves_dataset_urls` — a full spec whose `data` and `hf_dataset` are URLs comes out of `_normalize_spec` as repo ids.

```
$ python tests/test_training_runner.py -v
Ran 15 tests in 0.042s

OK
```

(The 13 pre-existing tests still pass unchanged.)

## End-to-end repro

A Studio run spec with the URL in `data`:

```json
{ "model": "Qwen/Qwen3-0.6B",
  "data": "https://huggingface.co/datasets/HuggingFaceH4/no_robots",
  "train_mode": "sft", "iters": 1, "test": true }
```

Before this change:

```
FileNotFoundError: Couldn't find any data file at /…/https:/huggingface.co/datasets/HuggingFaceH4/no_robots.
```

After this change the run completes and writes the adapter:

```
Iter 1 | Test | test loss 3.337
[Studio] Run complete. Outputs saved under …/adapters
```

## Note on the Swift change

`addCustomHFAsset` is also used for models. Model URLs are normalised the same way, because the same "interpreted as a local path" failure applies there.
