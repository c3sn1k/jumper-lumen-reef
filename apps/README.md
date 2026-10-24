# Simulator bundles

| App | Policy | Notes |
|---|---|---|
| `stairs_final.app` | `model_2199.pt` / `model_2200.pt` | Final checkpoint, 2,200-iteration run, Curriculum Level 3 |
| `stairs_1000.app` | `model_1000.pt` | Iteration 1,000, the point where Level 3 was reached |

Both are standard Jumper policy bundles: a ZIP whose root holds `bundle.json`,
`controller.toml`, `stairs.json`, `reference.json`, the ONNX policy under `models/`,
and the controller runtimes under `runtime/`.

## Load in the official simulator

Choose the simulator's bundle / `.app` upload and load either file. The bundle selects
`jumper.stairs`, builds the Lumen Temple world, and drives the simulated servos with the
controller's joint targets.

## Load locally

```bash
# from a checkout of KingKongRobotics/jumper
python scripts/play.py --app /path/to/apps/stairs_final.app
```

Controls are the bundle's own: left stick walk/strafe, right stick turn, `W/A/S/D` or
arrow keys, `J/L` turn, `Esc` release, plus gamepad support.

## Scope

These bundles are simulator builds. They contain ONNX and the web/native controller, but
no `.rknn` and no board runtime, so they are not intended for the physical robot. Rebuild
with the full Docker/Rust/RKNN toolchain and no `--allow-incomplete` for a board bundle.
