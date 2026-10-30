# Training replay clips

| File | Checkpoint | Setup |
|---|---|---|
| `model_1000_training.mp4` | `checkpoints/model_1000.pt` (recommended) | 15 s, `temple_stairs`, 200 Hz, 640x480, 30 fps |
| `model_2200_training.mp4` | `checkpoints/model_2199.pt` / `model_2200.pt` (for comparison) | 15 s, `temple_stairs`, 200 Hz, 640x480, 30 fps |

Both clips are replays of the exported policy in the training environment, recorded with
the upstream `scripts/play.py` (the same replay path the Colab notebook uses). They are
**not** web-simulator screen captures.

For display, the reset is pinned to the default start pose - at the base of the
staircase, yaw 0 - because training normally randomises the robot's `x`/`y` offset and
`yaw` on every episode. The command is 0.5 m/s forward. The policy and checkpoint are
unchanged.
