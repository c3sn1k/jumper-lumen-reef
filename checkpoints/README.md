# Checkpoints

All files come from the same training run:

```
logs/jumper/jumper.stairs/2026-10-19_12-59-54/
```

| File | Iteration | Size | SHA-256 |
|---|---|---|---|
| `model_1000.pt` | 1,000 | 10,383,669 bytes | `A7DBB01035E595C954DBE4E40863C7411BB51D197D91ED7DD15D16E61A5AD29C` |
| `model_2199.pt` | 2,199 (trainer's last file) | 10,383,669 bytes | `F7BB73585FED82E433A2823B26D79571156F6B5B8C75092FAF39AC886558BA7A` |
| `model_2200.pt` | alias of 2,199 | 10,383,669 bytes | `F7BB73585FED82E433A2823B26D79571156F6B5B8C75092FAF39AC886558BA7A` |

The run is described as **2,200 iterations**. Checkpoint names are zero-indexed, so the
trainer wrote `model_2199.pt` as its last file. `model_2200.pt` is included under the
requested name as a byte-identical copy of that final checkpoint; it is not a different
training step.

## Mapping to the bundles

| Checkpoint | Bundle |
|---|---|
| `model_1000.pt` | `apps/stairs_1000.app` |
| `model_2199.pt` / `model_2200.pt` | `apps/stairs_final.app` |

Each bundle contains the ONNX exported from the matching checkpoint (`models/stairs.onnx`)
and the deployment contract (`stairs.json`). To rebuild an `actor.onnx` from a `.pt`,
use the upstream `scripts/export.py` in the `jumper` repository, or inspect the ONNX
already inside the bundle.
