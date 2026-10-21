## Problem

The repo already has a Python test suite — `Tests/test_training_runner.py` — but nothing ever runs it automatically. `.github/workflows/` only contains `release.yml`, which is tag-driven, so backend regressions are caught only when a developer remembers to run the suite by hand.

## Change

Adds `.github/workflows/tests.yml`:

- runs on pushes to `main`, on every pull request targeting `main`, and on manual dispatch;
- one `backend` job on `macos-15` that runs `python Tests/test_training_runner.py -v`.

The suite stubs `mlx`, `mlx_lm` and `mlx_lm_lora`, so the job needs nothing but `numpy` — no mlx wheel download, no `mlx-lm-lora` install. It finishes in about a second.

## Local verification

Simulated the CI environment exactly (a clean virtualenv with only `numpy` installed, no mlx):

```
$ python Tests/test_training_runner.py -v
Ran 15 tests in 0.147s

OK
```

## Note

The Swift test target (`Tests/MLXLoRAStudioTests`) is not wired up here on purpose — it needs a full Xcode toolchain, and I would rather land a job I have actually run than one I have not. A separate `swift test` job can follow once it is verified on a runner.
