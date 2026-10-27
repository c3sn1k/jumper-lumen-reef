# Lumen Reef design assets

| File | What it is |
|---|---|
| `lumen-temple.map` | Built Lumen Temple map with the Lumen Reef skin embedded |
| `lumen-temple.scene.json` | Source scene for the staircase, altar and treasure |
| `lumen-reef.skin` | Standalone bioluminescent Lumen Reef shell |
| `lumen-temple-light.png` | Light look of the map (cyan walls and floor), matching the final capture |
| `lumen-temple-night.png` | Authored night look of the same map |
| `lumen-temple-path.png` | Camera-path preview (night look) |
| `lumen-reef-skin.png` | Shell preview |
| `design-brief.md` | Original design intent, silhouette and palette |
| `selection.json` | Selected concept and user confirmation |

The map has two looks:

- **Light** - cyan walls and floor, the appearance used in the final video and the
  standard web-simulator look. Same geometry and staircase.
- **Night** - the authored bioluminescent look stored in the map package: aqua fog, cyan
  glows and the floating gold treasure under a dark sky.

The night look is the one embedded in `lumen-temple.map`; the web simulator presents the
same map with its standard lighter appearance, which is what the capture used. Both
previews are included so the repository does not imply only one look exists.

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
