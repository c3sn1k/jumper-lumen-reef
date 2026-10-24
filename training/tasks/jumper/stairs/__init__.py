"""jumper.stairs -- tripod gait trained on a 0.03-0.06 m temple staircase.

The staircase is a physical prop built by this task (see
`scenes/temple_stairs.py`), so the task always trains on it; `--scene
temple_stairs` only changes the look. The tripod skeleton supplies the gait and
the velocity reward. This task adds height progress, stronger foot-lift shaping
and a backward/tilt penalty, and raises the foot target to 0.06 m.

This is a terrain/apparatus variant of `jumper.tripod`, not a fifth gait, so it
is not one of the four locomotion controls in `tests/test_task_parity.py`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.stairs",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod climbing a ten-step temple staircase, 0.03-0.06 m rises; "
        "tripod gait, height-progress and foot-clearance shaping"
    ),
    tags=("locomotion", "jumper", "stairs", "climb", "tripod"),
)
