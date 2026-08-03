# Compliant Surface Locomotion

Research code and training scripts built on top of
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) (kit-less / Newton backend).

This repo is a [uv](https://docs.astral.sh/uv/) project (Python 3.12). Isaac Lab
itself is **not** a locked dependency here — clone and install it into this
project’s `.venv` separately.

## Requirements

- Linux with a CUDA-capable GPU
- [uv](https://docs.astral.sh/uv/)
- Isaac Lab tag **`v3.0.0-beta2.patch1`** (kit-less install)

Kit-less install docs:
https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/setup/installation/kitless_installation.html

## Setup

### 1. Sync the uv project environment

```bash
cd compliant-surface-locomotion
uv sync
```

This creates `.venv` from `pyproject.toml` / `uv.lock` (Python 3.12).

### 2. Clone and install Isaac Lab into the same venv

```bash
cd ..
git clone https://github.com/isaac-sim/IsaacLab.git --branch v3.0.0-beta2.patch1
cd IsaacLab

# Point the install at this project's venv, then install kit-less extras
source ../compliant-surface-locomotion/.venv/bin/activate
./isaaclab.sh -i 'newton,rl[rsl-rl],visualizer[newton]'
```

Keep Isaac Lab as a separate checkout. Pin the tag above when you upgrade.

**Important:** plain `uv sync` removes packages that are not in `uv.lock`, which
will wipe Isaac Lab. After Isaac Lab is installed, either avoid bare `uv sync`,
or re-sync with:

```bash
uv sync --inexact
```

### 3. Verify Isaac Lab is usable from this repo

```bash
cd ../compliant-surface-locomotion
uv run python scripts/verify_isaaclab.py
```

You should see `OK` lines for `isaaclab`, `isaaclab_newton`, and `isaaclab_rl`.

## Day-to-day usage

```bash
uv run python scripts/verify_isaaclab.py

# Stand ANYmal-C under a joint-space PD controller, shown in the Newton viewer
uv run python scripts/view_anymal.py
```
