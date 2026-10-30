# Pedestal - Colab / training overlay

This overlay reproduces the **attractor** pedestal run whose checkpoints are in
[`../checkpoints/`](../checkpoints/). It mirrors the upstream
[`KingKongRobotics/jumper`](https://github.com/KingKongRobotics/jumper) tree and
must be copied over a fresh checkout; it is not a standalone repository.

## 1. Apply the overlay

```bash
# from the root of a fresh KingKongRobotics/jumper checkout
cp -r /path/to/lumen-reef-challenge/pedestal/training/tasks/jumper/pedestal tasks/jumper/pedestal
cp /path/to/lumen-reef-challenge/pedestal/training/scenes/pedestal.py scenes/pedestal.py
cp /path/to/lumen-reef-challenge/pedestal/training/tasks/__init__.py tasks/__init__.py
cp /path/to/lumen-reef-challenge/pedestal/training/tools/warmstart_pedestal.py tools/warmstart_pedestal.py
cp /path/to/lumen-reef-challenge/pedestal/training/tools/colab.py tools/colab.py
```

`tasks/jumper/pedestal/env_cfg.py` carries the **environment contract**: 22
degrees of freedom, `nconmax = 2000`, `njmax = 5000`, the pedestal geometry and
the contact sensors. See [`../README.md`](../README.md) for the full table.

## 2. Warm start from the stairs actor

```bash
python tools/warmstart_pedestal.py \
  --source checkpoints/model_1000.pt \
  --out /content/pedestal_warmstart.pt
```

The stairs checkpoint (`model_1000.pt`) is in the repository root. The script
maps the stairs actor (411 obs / 20 actions) into the pedestal layout
(451 obs / 22 actions), initialises the two gripper columns/rows and writes an
iteration-0 checkpoint. The critic and optimiser are fresh.

## 3. Train

```bash
python scripts/train.py \
  --task jumper.pedestal --model jumper \
  --backend warp --device cuda:0 \
  --num_envs 256 --max-iterations 10000 \
  --headless --logger tensorboard \
  --resume --checkpoint /content/pedestal_warmstart.pt
```

The height curriculum visits 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22 and
0.24 m, 4,000 control steps per rung. At each rung the reset samples the exposed
step uniformly in `[0.08, rung]`. Level 8 (the full range) is reached at about
iteration 1,333; the rest of the run trains the whole range.

## Reward terms (the attractor payload)

| Term | Weight | Note |
|---|---:|---|
| `claw_height_contact` | **+8.0** | per-claw height while contacting the pedestal, until mounted |
| `claw_top_anchor` | **+15.0** | **both claws** on the top rim, until mounted |
| `anchored_leg_lift` | **+2.5** | **both claws** anchored; lift the six feet, minus a per-foot penalty for feet below the rim |
| `full_mount_success` | **+20.0** | root inside `0.5 * radius` at standing height for a 10-step dwell |
| `claw_attractor` | **+6.0** | product of two Gaussians on the claw tips toward the **centre** of the top, `scale = 0.25`; after contact, until mounted |
| `stagnation_penalty` | **-5.0** | x progress before contact, z progress after; 50-step patience |
| `handstand_kill_switch` | **-20.0** | both front claws down + a rear foot lifted |
| `forward_lean_penalty` | **-5.0** | nose-down pitch |
| `wall_hug_pitch` | **+1.5** | backward pitch near the wall |
| `upright` / `pose` | **0 at the wall** | gated for `root_x > -0.15` |
| `stance_load_stand` | **0.0** | disabled |

After the first claw-pedestal contact, `track_linear_velocity`,
`track_angular_velocity`, `tripod_gait`, `stance_load`, `foot_clearance`,
`foot_swing_height` and `stance_ground_gap` are switched off by the contact latch.
The attractor is the dense post-contact term; the jackpot and the guards stay on.

## 4. Replay

```bash
python scripts/play.py \
  --task jumper.pedestal --model jumper \
  --checkpoint /path/to/lumen-reef-challenge/pedestal/checkpoints/model_3000.pt \
  --pedestal-height 0.08 \
  --backend warp --device cuda:0 \
  --num_envs 1 --headless --steps 750 \
  --video pedestal_3000.mp4
```

Drop `--headless`/`--video` for the live viewer. Without `--pedestal-height`,
replay uses the final 0.24 m rung; that is the rung the mechanism cannot mount.
Use 0.08-0.18 for the checkpoints that still work, starting with `model_3000.pt`
at 0.08 m. `--command 0.2 0 0` holds a forward approach; without it the operator
keyboard/pad drives.

## 5. Known limit

The 0.20-0.24 m rungs are beyond the mechanism: the motors saturate and the legs
do not have enough leverage. The run spent most of its tail there, so the late
checkpoints lean on the rim instead of mounting. This is why the low-height
payloads (`0.08-0.18 m`) exist; see the repository README for the current
recommendation.
