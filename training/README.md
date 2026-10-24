# Training overlay

This directory mirrors the upstream
[`KingKongRobotics/jumper`](https://github.com/KingKongRobotics/jumper) tree. It is an
overlay, not a standalone framework: put these files over a fresh checkout of that
repository, then follow its setup guide.

## Files and why they are here

| Path | Change |
|---|---|
| `tasks/jumper/stairs/` | New registered task `jumper.stairs` (env, rewards, PPO config, controls) |
| `tasks/jumper/tripod/env_cfg.py` | Accepts `foot_target_height` (default 0.025 m, so tripod is unchanged) |
| `scenes/temple_stairs.py` | Staircase geometry (0.03-0.06 m rises, 0.30 m run, 1.8 m wide, 0.44 m top) and training look |
| `scenes/__init__.py` | Registers `temple_stairs` |
| `tasks/__init__.py` | Imports the new task |
| `tests/test_task_parity.py` | Accounts for `jumper.stairs` as a tripod variant |
| `notebooks/jumper_colab.ipynb` | Adds `jumper.stairs`, the `temple_stairs` scene and replay defaults |
| `tools/colab.py` | Accepts `--scene` |
| `COLAB_STAIRS.md` | Exact Colab, replay and export commands |

## Overlay

Run from the root of a fresh `jumper` checkout:

```bash
cp -r  /path/to/lumen-reef-challenge/training/tasks/jumper/stairs tasks/jumper/stairs
cp     /path/to/lumen-reef-challenge/training/tasks/jumper/tripod/env_cfg.py tasks/jumper/tripod/env_cfg.py
cp     /path/to/lumen-reef-challenge/training/scenes/temple_stairs.py scenes/temple_stairs.py
cp     /path/to/lumen-reef-challenge/training/scenes/__init__.py scenes/__init__.py
cp     /path/to/lumen-reef-challenge/training/tasks/__init__.py tasks/__init__.py
cp     /path/to/lumen-reef-challenge/training/tests/test_task_parity.py tests/test_task_parity.py
cp     /path/to/lumen-reef-challenge/training/tools/colab.py tools/colab.py
cp     /path/to/lumen-reef-challenge/training/notebooks/jumper_colab.ipynb notebooks/jumper_colab.ipynb
```

Then train:

```bash
python tools/colab.py train \
  --repo . --task jumper.stairs --model jumper \
  --num-envs 256 --iterations 2200 \
  --manifest session/stairs.json --scene temple_stairs
```

See [`COLAB_STAIRS.md`](COLAB_STAIRS.md) for the notebook workflow, replay commands and
the `.app` export procedure, including the `--allow-incomplete` browser/simulator build.
