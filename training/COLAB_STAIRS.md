# Lumen Temple — stairs training run

The map is built and verified; the training task and reward are in the repo; the
Colab notebook is updated. This file lists what changed and the exact commands.

## Files changed / added

Design repo (`jumper-design`):

- `workspaces/lumen-reef/map/lumen-temple.scene.json` — 10-step staircase, altar,
  floating gold treasure, retained aqua bioluminescence/fog.
- `workspaces/lumen-reef/delivery/lumen-temple.map` — verified map (65 files,
  spawn OK, skin `lumen-reef` included).
- `workspaces/lumen-reef/map/lumen-temple-cam_path.png` — map thumbnail.

The staircase was raised after the first simulator test: the rises are now
`0.03, 0.03, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.06, 0.06`, total 0.44 m.
The earlier map's first four steps (0.02/0.02/0.03/0.03) were climbable with
normal walking; the new first steps start at 0.03 and should demand the trained
policy earlier. Re-test the new `lumen-temple.map`.

Training repo (`jumper-main`):

- `scenes/temple_stairs.py` — staircase geometry (0.03→0.06 m rises, 0.30 m run,
  1.8 m wide, 0.44 m top) and the look-only training scene.
- `scenes/__init__.py` — registers `temple_stairs`.
- `tasks/jumper/tripod/env_cfg.py` — accepts `foot_target_height` (default stays
  0.025 m, so tripod is unchanged).
- `tasks/jumper/stairs/__init__.py` — registers `jumper.stairs`.
- `tasks/jumper/stairs/env_cfg.py` — installs the staircase prop, reset event,
  height-progress reward, backward/tilt penalty and stronger foot-clearance
  penalties; foot target 0.06 m.
- `tasks/jumper/stairs/mdp/rewards.py` — `climb_height` and `backward_lean`.
- `tasks/jumper/stairs/rl_cfg.py` — tripod PPO baseline, `jumper.stairs` log name.
- `tasks/jumper/stairs/controls.yaml` — same operator controls as tripod.
- `tasks/__init__.py` — imports the task.
- `tests/test_task_parity.py` — accounts for `jumper.stairs` as a tripod variant.
- `notebooks/jumper_colab.ipynb` — adds `jumper.stairs`; training adds
  `--scene temple_stairs`; replay defaults to `temple_stairs`.
- `tools/colab.py` — accepts `--scene` (already patched for `jumper.climb`).

Payload for Colab: `colab-stairs-files.zip` (overlay on `/content/jumper`).

## Reward summary

On top of the tripod velocity skeleton:

- `track_linear_velocity` (existing) — forward speed tracking.
- `climb_height` weight **+1.5** — normalised root-height progress from 0.10 m
  to 0.39 m (the new staircase top is 0.44 m).
- `backward_lean` weight **−2.0** — stronger for backward base velocity, plus
  squared tilt beyond 0.35 of the projected-gravity horizontal.
- `foot_clearance` weight **−3.0** and `foot_swing_height` weight **−0.5**,
  with the foot target raised to **0.06 m**.
- `upright` (existing) remains; the actor stays proprioceptive only, so the
  policy is deployable in the current `.app`.

The 0.06 m top step is above the hardest rough-terrain step used before
(0.048 m), so expect a real struggle. If it does not learn, lower the last rises
to 0.05 m (`scenes/temple_stairs.py` `RISES` and the map steps must agree).

Measured on the previous 0.02/0.02/0.03/0.03 staircase: the first four steps were
climbed with normal walking. That map has been replaced; the new first rises are
3-4 cm, so re-test `lumen-temple.map` before assuming where normal walking stops.
The last six rises (4-6 cm) are the training target.

## Colab

1. Upload `jumper-main/notebooks/jumper_colab.ipynb` to Colab and select a GPU.
2. Run the setup cell (clones `/content/jumper` and installs).
3. Upload `colab-stairs-files.zip` to `/content`, then run:

   ```python
   !unzip -o /content/colab-stairs-files.zip -d /content/jumper
   ```

4. In the run settings choose **TASK = jumper.stairs**, **NUM_ENVS = 256**.
   Start with **ITERATIONS = 2200**. The training cell adds
   `--scene temple_stairs` automatically.
5. Run training. If the runtime drops, choose **Continue training** with the
   backup; the task must stay `jumper.stairs`.

Manual equivalent of the training cell (inside `/content/jumper`):

```bash
python tools/colab.py train \
  --repo /content/jumper --task jumper.stairs --model jumper \
  --num-envs 256 --iterations 2200 \
  --manifest /content/session/stairs.json --scene temple_stairs
```

## MP4 replay

`SIM_SCENE` defaults to `temple_stairs` for this task. Set
`SIM_SECONDS = 25`, `SIM_COMMAND = "Forward"` (or Sampled), then run the replay
cell. Manual equivalent:

```bash
python scripts/play.py \
  --task jumper.stairs --model jumper \
  --checkpoint <path>/model_<N>.pt \
  --scene temple_stairs --backend warp --device cuda:0 \
  --num_envs 1 --headless --steps 1250 \
  --video /content/stairs.mp4
```

1250 control steps at 50 Hz is 25 s. The MP4 is written by `scripts/play.py`.

## .app export

Two steps, after a checkpoint exists. First build the deployment export:

```bash
python scripts/export.py \
  --task jumper.stairs --model jumper \
  --checkpoint <path>/model_<N>.pt \
  --out tasks/jumper/stairs/out/<run>
```

Then build the app. With a full toolchain (Rust, wasm-bindgen 0.2.128, Docker for
the board/RKNN):

```bash
python scripts/deploy.py \
  --mode stairs=tasks/jumper/stairs/out/<run> \
  --bundle out/app-stairs
```

For a browser/video build on a machine without Docker, `--allow-incomplete` skips
the cross-builds and RKNN conversion but still needs `cargo` and wasm-bindgen:

```bash
python scripts/deploy.py \
  --mode stairs=tasks/jumper/stairs/out/<run> \
  --bundle out/app-stairs --allow-incomplete
```

The one-off bundle is named `bundle`: the directory is `out/app-stairs/bundle/`
and the file is **`out/app-stairs/bundle.app`** (the command prints the exact
paths). The board's `.rknn` and the board controller are only present when the
Docker cross-build ran, so do not treat an `--allow-incomplete` app as ready for
the robot.

## Competition notes

- The map and the training staircase share the same rises, run and width.
- The map's altar platform is at 0.40 m; the floating gold sphere is the
  treasure. The environment stays aqua/cyan with the existing fog.
- If the policy cannot clear 0.06 m, the honest fallback is 0.05 m; do not claim
  a climb the replay does not show.
