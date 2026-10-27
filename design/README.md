# Lumen Reef design assets

| File | What it is |
|---|---|
| `lumen-temple.map` | Built Lumen Temple map with the Lumen Reef skin embedded |
| `lumen-temple.scene.json` | Source scene for the staircase, altar and treasure |
| `lumen-reef.skin` | Standalone bioluminescent Lumen Reef shell |
| `lumen-temple-light.png` | Frame from the official web simulator: the azure look the simulator gives the map |
| `lumen-temple-night.png` | Authored night look stored in the map (dark) |
| `lumen-temple-path.png` | Camera-path preview (night look) |
| `lumen-reef-skin.png` | Shell preview |
| `design-brief.md` | Original design intent, silhouette and palette |
| `selection.json` | Selected concept and user confirmation |

The map has two looks:

- **Web simulator / light** - the map as the official web simulator renders it: light
  azure walls and floor. This preview is a frame from the simulator, not a native render.
- **Night** - the authored bioluminescent look stored in `lumen-temple.map`: dark sky,
  aqua fog, cyan glows and the floating gold treasure.

The night look is the one embedded in the map file. The web simulator re-renders the same
map with its standard light appearance, which is what the capture used; the light image
below is a simulator frame. Both previews are included so the repository shows both the
authored dark map and the light look used in the capture.

The map pairs with the staircase used in training:

- ten steps, rises `0.03, 0.03, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.06, 0.06` m,
- 0.30 m run, 1.8 m width, total top 0.44 m,
- altar platform at 0.40 m, floating gold treasure, aqua fog and cyan bioluminescence.

The assets were authored with the upstream
[`KingKongRobotics/jumper-design`](https://github.com/KingKongRobotics/jumper-design)
workflow. Copy the `.map` and `.skin` into that workflow's content library, or load the
map directly in the official simulator.

License and provenance: see the repository [`LICENSE`](../LICENSE) and
[`NOTICE`](../NOTICE).
