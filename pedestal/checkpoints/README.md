# Pedestal checkpoints

These are the useful subset of the attractor run kept in
`backup_pedesta_final_training_10000` (10,000 PPO iterations, `jumper.pedestal`,
256 environments). The run saved 50 files; this repository publishes the ones
worth trying, not the whole sweep.

| File | Iteration | Step counter | Curriculum level | Bytes | SHA-256 |
|---|---:|---:|---:|---:|---|
| `model_3000.pt` | 3,000 | 72,024 | 8 | 10,877,365 | `75d3ad9728bc5203...` |
| `model_5000.pt` | 5,000 | 120,024 | 8 | 10,877,365 | `f0e6d61d262ba9b9...` |
| `model_9800.pt` | 9,800 | 235,224 | 8 | 10,877,365 | `a7b4c8eb5d882170...` |
| `model_9999.pt` | 9,999 | 240,000 | 8 | 10,877,365 | `bca2d1466f284777...` |

Full hashes are in [`SHA256SUMS`](SHA256SUMS). Checkpoint names are zero-indexed.
The curriculum level is `min(8, counter // 4000)`: one rung per 4,000
per-environment control steps (24 steps per PPO iteration). All four files
here are from level 8, the phase that trains the whole [0.08, 0.24] range.

## The published subset

- `model_3000.pt` - best measured at **0.08 m**: the body reaches the pedestal
  footprint and stays for long stretches (body at the rim, rear feet on the
  platform); not the inner-half jackpot.
- `model_5000.pt` - next best, at **0.10 m**.
- `model_9800.pt` - late checkpoint tested at **0.18/0.22 m**; it hooks the
  rim and leans, without a clean mount.
- `model_9999.pt` - the final file of the run, from the same late phase as
  9800.

## Measured notes (local replay) on native CPU

Replayed at a **fixed** exposed step, the run does not complete a clean mount
on the 0.20-0.24 m rungs: it hooks the rim and leans, and at 0.24 it also
falls. The useful range is the low one (0.08-0.18 m). At 0.24 m the mechanism
cannot make the last step; the run spent its tail there, which is why the
late checkpoints are not better at the low rungs.

The local notes are `native`/CPU; the official replay is `warp`/GPU and may
differ slightly. Treat these four as a small sweep and pick the one that
works for the height you are testing.

## Warm-start source

The stairs checkpoint used as the actor initialisation (`model_1000.pt`) is
in the repository root at [`checkpoints/model_1000.pt`](../../checkpoints/model_1000.pt),
not in this directory.
