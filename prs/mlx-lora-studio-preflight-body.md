## Problem

When the selected Python environment does not have the MLX runtime installed, the app fails with a bare traceback from the middle of the import block:

```
$ /usr/bin/python3 Backend/training_runner.py --spec run_spec.json
Traceback (most recent call last):
  File ".../Backend/training_runner.py", line 20, in <module>
    import mlx.core as mx
ModuleNotFoundError: No module named 'mlx'
```

The message names neither the interpreter that was used nor the fact that the fix is to pick a different environment or install the packages, so the natural reaction is "the app is broken".

This is the default state of a fresh macOS install, because `/usr/bin/python3` (the Command Line Tools interpreter) has no `mlx`.

## Fix

Add a preflight check that runs before the MLX imports:

```
$ /usr/bin/python3 Backend/training_runner.py --spec run_spec.json
MLX LoRA Studio cannot start: mlx, mlx_lm, mlx_lm_lora are not installed in
  /Library/Developer/CommandLineTools/usr/bin/python3

Install the runtime packages with:
  /Library/Developer/CommandLineTools/usr/bin/python3 -m pip install mlx mlx-lm-lora matplotlib

or choose a different environment in Settings > Python environments.
```

and exits with status `2`.

The check looks for `mlx`, `mlx_lm` and `mlx_lm_lora` — the three packages `Backend/requirements.txt` installs — so it also catches a partially provisioned environment (e.g. `mlx` present but `mlx-lm-lora` missing), which currently fails later and further from the cause.

`importlib.util.find_spec` is not used for modules that are already in `sys.modules`: the test suite injects stubs for them, and a stub module has no `__spec__` for `find_spec` to return.

## Tests

Two new cases in `Tests/test_training_runner.py`:

- `test_preflight_passes_when_the_runtime_is_available` — nothing is reported missing when the runtime is importable.
- `test_preflight_reports_the_missing_runtime_modules` — all three packages are reported when `find_spec` finds nothing.

```
$ python Tests/test_training_runner.py -v
Ran 15 tests in 0.036s

OK
```
