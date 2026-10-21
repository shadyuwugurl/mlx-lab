## Problem

The test-split evaluation ran unguarded inside `run()`:

```python
# Backend/training_runner.py
studio_log("Evaluating test split")
test_output = TeeCapture()
with contextlib.redirect_stdout(test_output), contextlib.redirect_stderr(test_output):
    evaluate_model(...)
```

Any exception raised inside `evaluate_model` — a malformed judge output, a tokenizer edge case, an out-of-memory in the reference model — propagates out of `run()`, aborts the job, and **the adapter that had just been trained is never written to disk**. The export step (`save_pretrained_merged(...)`, and the `mx.clear_cache()` that precedes it) comes *after* the evaluation, so a failed *optional* metric destroys the *required* output.

This is easy to hit: the evaluation runs on the reference/judge models, which are the largest objects in the job.

## Fix

Move the evaluation into a helper that reports failures instead of raising them:

```python
def _evaluate_test_split(...) -> dict[str, float]:
    test_output = TeeCapture()
    try:
        with contextlib.redirect_stdout(test_output), contextlib.redirect_stderr(test_output):
            evaluate_model(...)
    except Exception as exc:
        studio_log(f"Test split evaluation failed: {type(exc).__name__}: {exc}")
        studio_log("Continuing without test metrics; the adapter will still be saved.")
        return {}
    return _test_metrics_from_output(test_output.buffer.getvalue(), args.iters)
```

`run()` then keeps its existing "only report metrics when there are some" behaviour, and the export step always runs.

The exception is caught broadly on purpose: the evaluation is an optional extra, and there is no useful distinction between "the judge model returned something unexpected" and "the reference model did not fit in memory" — both mean "no test metrics this time", not "abort the run".

## Tests

Two new cases in `Tests/test_training_runner.py`:

- `test_failed_test_evaluation_is_reported_not_raised` — `evaluate_model` raises; the helper returns `{}` instead of propagating.
- `test_successful_test_evaluation_returns_metrics` — the helper still parses `Loss` / `Perplexity` / `Tokens` out of the captured output.

```
$ python Tests/test_training_runner.py -v
Ran 15 tests in 0.036s

OK
```
