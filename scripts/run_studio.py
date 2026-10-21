#!/usr/bin/env python3
"""Run a Studio training spec through the installed app's backend.

The app's own ``main()`` swallows tracebacks, so this loads the backend module
directly and prints the exception.  ``--backend`` selects the patched app
bundle (default) or the upstream repository checkout.
"""
import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

APP_BACKEND = "/Applications/MLX LoRA Studio.app/Contents/Resources/StudioSupport/Backend/training_runner.py"
REPO_BACKEND = "/Users/charlie/Documents/mlx-lab/upstreams/MLX-LoRA-Studio/Backend/training_runner.py"

# The app puts its bundled vendored mlx-lm-lora on PYTHONPATH for the child
# processes it spawns; running the backend by hand has to do the same, or
# ``mlx_lm_lora`` is not importable.  NOTE: the vendor checkout lives directly
# under ``vendor/`` and the package dir is ``vendor/mlx_lm_lora`` (no "-lora").
APP_VENDOR = "/Applications/MLX LoRA Studio.app/Contents/Resources/StudioSupport/vendor"


def load_backend(path):
    spec = importlib.util.spec_from_file_location("studio_training_runner", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["studio_training_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--backend", default=APP_BACKEND, choices=[APP_BACKEND, REPO_BACKEND])
    args = ap.parse_args()

    if args.backend == APP_BACKEND:
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [APP_VENDOR] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
        )

    mod = load_backend(args.backend)
    with open(args.spec) as fh:
        raw = json.load(fh)
    try:
        mod.run(mod._normalize_spec(raw))
    except SystemExit as exc:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
