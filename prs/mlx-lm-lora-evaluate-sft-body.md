## Summary

`evaluate_sft()` returns only the loss, but `mlx_lm_lora.train` unpacks **two** values from it when it reports the test split:

```python
test_loss, test_ntokens = evaluate_sft(...)
```

That raises `TypeError: cannot unpack non-iterable float object`, so every SFT run with a non-empty test split **crashes after the training loop has already finished**, and the caller never receives the test loss / perplexity it asked for.

The mismatch is present in `v3.1.2`, in `v5.8.6` and on `main` (`fd3b2dd`).

## Reproduce

Any SFT job that evaluates a test split:

```
mlx_lm_lora.train --train-mode sft --data <dir containing test.jsonl> --test
```

fails at the reporting step with:

```
TypeError: cannot unpack non-iterable float object
```

The traceback points at `mlx_lm_lora/train.py`, in the `elif args.train_mode == "sft":` branch, after `train_sft(...)` has already returned.

## Fix

`evaluate_sft()` now returns `(loss, n_tokens)`, matching what every caller already expects. The two callers that only want the loss (`trainer/sft_trainer.py` for validation, `train_judge.py` for judging) unpack and discard the token count.

## Test

`tests/test_evaluate_sft.py` is a new regression test. It fails on the unpatched code with the same `TypeError` and passes with the fix:

```
$ python -m pytest tests/test_evaluate_sft.py -q
..                                                                       [100%]
2 passed
```

Full suite: `151 passed, 56 subtests passed`.

## Note for maintainers

`evaluate_sft` is also reached through `mlx_lm_lora.train_judge` and from MLX-LoRA-Studio, which vendors this package. There the same crash aborts the Studio run before its adapter export step, so a completed fine-tune is reported as a failure.
