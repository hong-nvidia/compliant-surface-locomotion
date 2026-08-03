#!/usr/bin/env python3
"""Smoke-test that Isaac Lab is importable in the active environment.

Run from the repo root after following the README setup:

    python scripts/smoke_test_isaaclab.py
"""

from __future__ import annotations


def _check(module_name: str) -> None:
    try:
        module = __import__(module_name)
    except ImportError as exc:
        print(f"FAIL  {module_name}: {exc}")
        raise SystemExit(1) from exc
    location = getattr(module, "__file__", "<namespace>")
    print(f"OK    {module_name}: {location}")


def main() -> None:
    # Core framework — required for any Isaac Lab workflow.
    _check("isaaclab")

    # Typical kit-less Newton extras from:
    #   ./isaaclab.sh -i 'newton,rl[rsl-rl],visualizer[newton]'
    _check("isaaclab_newton")
    _check("isaaclab_rl")

    print("Isaac Lab smoke test passed.")


if __name__ == "__main__":
    main()
