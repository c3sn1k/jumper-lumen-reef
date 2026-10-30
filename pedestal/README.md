# Pedestal task - `jumper.pedestal`

The crab walks off the platform edge, hooks the top rim of a round pedestal with
its two front claws, and pulls the body up onto it. This directory publishes the
checkpoints of the most successful pedestal attempt and the environment contract
they were trained against.

Unlike the staircase, this is not a gait: it is a mounting task. The actor stays
proprioceptive; it finds the pedestal by contact.

## The method

The best result came from the **attractor payload**:

1. **Warm start** - the stairs actor (`checkpoints/model_1000.pt`, 411 obs / 20
   actions) is mapped into the pedestal layout (451 obs / 22 actions). The four
   per-joint observation blocks are expanded, the two gripper columns start at
   zero, and the two action rows are initialised so the grippers explore from the
   first step. The critic and optimiser are **fresh**.
2. **Contact phase machine** - after the first claw-pedestal contact the
   locomotion rewards (velocity tracking, gait, foot terms) stop; the claw
   attractor and the pull-up cascade drive the mount. The latch is sticky for the
   episode.
3. **Height curriculum** - nine rungs from 0.08 m to 0.24 m, 4,000 control steps
   per rung. At each rung the reset samples the exposed step uniformly in
   `[0.08, rung]`, so the final policy is trained across the whole range.
4. **Reward cascade** - a dense attractor pulls the claw tips toward the pedestal
   top; the bi-manual rim anchor and leg-lift terms pay the pull-up; a full-mount
   jackpot pays the mounted pose after a 10-step dwell. The exact terms are in
   [`training/tasks/jumper/pedestal/mdp/rewards.py`](training/tasks/jumper/pedestal/mdp/rewards.py).

The run is **10,000 PPO iterations** with 256 environments. The 50 checkpoints in
[`checkpoints/`](checkpoints/) are every file the trainer saved.

## Environment contract

These are the settings a checkpoint must be replayed with. They are not cosmetic:
the 22-joint robot drops contacts at the skeleton's default buffers and goes slack
at frame zero.

| Setting | Value |
|---|---|
| Task | `jumper.pedestal` |
| Model | `jumper` (`assets/jumper/jumper.xml`) |
| Degrees of freedom | **22** = 20 tripod gait joints + 2 grippers (`LF_J4`, `RF_J4`) |
| Action | 22 joint-position targets, `ACTION_SCALE` per joint |
| Actor observation | **451** (proprioceptive only) |
| Claw bodies | `LF_finger_link`, `RF_finger_link`; tips `LF_finger_tip_link`, `RF_finger_tip_link` |
| **Contact buffers** | `nconmax = 2000`, `njmax = 5000` |
| Control / physics | **50 Hz / 200 Hz** (`timestep = 0.005`, `decimation = 4`) |
| Episode | 20 s = 1,000 control steps |
| Platform | top at **0.44 m**, 2.4 x 1.8 m |
| Pedestal | cylinder radius **0.38 m**, body half-height 0.20 m, centre at x = 0.60 m |
| Exposed step | **0.08-0.24 m** (the body is placed lower than the platform so only the exposed part changes) |
| Claw contact sensor | `claw_pedestal_contact`: claw subtree vs `pedestal` body |
| Foot contact sensor | `feet_surface_contact`: foot geoms vs any surface |
| Command | forward `0.0-0.4 m/s`, lateral `+/-0.05`, yaw `+/-0.1 rad/s` |
| Reset | robot on the platform at x in `[-0.10, -0.05]`, y `+/-0.03`, yaw `+/-0.03`; pedestal at x = 0.60 |
| Warm start | actor only from the stairs `model_1000.pt`; critic and optimiser fresh |

The collision buffers are set twice because this framework has two simulators:
`cfg.sim.nconmax` / `cfg.sim.njmax` for warp, and `MjSpec.nconmax` / `njmax`
through `scene.spec_fn` for the native compile. The file that does this is
[`training/tasks/jumper/pedestal/env_cfg.py`](training/tasks/jumper/pedestal/env_cfg.py).

## Checkpoints

[`checkpoints/`](checkpoints/) holds all 50 files, iteration 150 to 9,999, with a
table (iteration, curriculum level at save, bytes, SHA-256) and a
[`SHA256SUMS`](checkpoints/SHA256SUMS) manifest. They are raw `.pt` files, not
`.app` bundles.

The measured behaviour, replayed locally on native CPU at a **fixed** exposed
step:

- `model_3000.pt` at **0.08 m** is the best measured: the body reaches the
  pedestal footprint and stays there for long stretches (body at the rim, rear
  feet on the platform). It is not the inner-half jackpot.
- `model_5000.pt` at **0.10 m** is the next best.
- `model_9800.pt` at **0.18 m** and 0.22 m hooks the rim and leans, without a
  clean mount.
- At 0.24 m the mechanism cannot make the last step (the motors saturate and
  there is not enough leverage). The run spent its long tail there, which is why
  the later checkpoints are not better on the low rungs.

The useful sweep is therefore **0.08-0.18 m**, with `model_3000.pt` the first
one to try. The local notes are `native`/CPU; the official replay uses
`warp`/GPU and may differ slightly.

## Replay

Apply the training overlay to a fresh `KingKongRobotics/jumper` checkout (see
[`training/COLAB_PEDESTAL.md`](training/COLAB_PEDESTAL.md)), then:

```bash
# fixed exposed step, one robot, record a video
python scripts/play.py \
  --task jumper.pedestal --model jumper \
  --checkpoint pedestal/checkpoints/model_3000.pt \
  --pedestal-height 0.08 \
  --backend warp --device cuda:0 \
  --num_envs 1 --headless --steps 750 \
  --video pedestal_3000.mp4
```

Drop `--headless` and `--video` for the live viewer. `--pedestal-height` fixes
the exposed step and disables the curriculum; without it, replay uses the final
0.24 m rung. `--command 0.2 0 0` holds a forward approach; otherwise the operator
keyboard/pad drives the velocity command. `--steps` is control steps at 50 Hz.

## Export and apps

**No pedestal `.app` is published.** The only bundle attempt was built from
`model_3999.pt`, which is a reward-hacked policy (it rears up on the spot to
maximise the root height) and is not included. The web host's deployment also
does not know `jumper.pedestal`, so a pedestal bundle needs its own controller/FSM
contract and gains.

The checkpoints are for replay and export. To export one:

```bash
python scripts/export.py --task jumper.pedestal --model jumper \
  --checkpoint pedestal/checkpoints/model_3000.pt --out out/pedestal_3000
```

then build a bundle with the deployment toolchain; expect the same policy
limitations described above.

## Reproduce the training

See [`training/COLAB_PEDESTAL.md`](training/COLAB_PEDESTAL.md). In short:
apply the overlay, build the warm-start checkpoint from the stairs
`model_1000.pt`, train `jumper.pedestal` for 10,000 iterations, and replay with
`--pedestal-height`.

## License and attribution

The pedestal task, scene and checkpoints are derivative work on the Jumper
framework by **KingKong Robotics**
([`KingKongRobotics/jumper`](https://github.com/KingKongRobotics/jumper),
Apache-2.0). See the repository [`LICENSE`](../LICENSE) and [`NOTICE`](../NOTICE).
