"""jumper.pedestal -- mounting a round pedestal with the front arms and claws.

This is not a fifth gait. It walks and climbs with the tripod skeleton, but the
action and observation include the two front grippers as well, and the task is a
single fixed apparatus: a platform at 0.44 m with a cylinder on it whose exposed
step is taught by a height curriculum.

The front arms are the reason the action is 22 joints instead of the locomotion
tasks' 20. The claws are what can hook the top edge; the four middle/rear legs
carry and push. The actor still sees only proprioception, so the pedestal is found
by contact, and the deployment contract can run the same policy through the
existing controller.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.pedestal",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod mounting a round pedestal; tripod skeleton with the two "
        "front arms and claws active, and a curriculum over the exposed step height"
    ),
    tags=("jumper", "pedestal", "climb", "manipulation", "tripod"),
)


def cli_args(group):
    """This task's own replay/training arguments.

    `--pedestal-height` fixes one exposed step and disables the height
    curriculum. It is the height-limit test: train or replay one checkpoint at
    0.10, 0.15, 0.20, 0.24 and see where the body actually stops mounting.
    """
    group.add_argument(
        "--pedestal-height",
        type=float,
        default=None,
        metavar="M",
        help="fix the pedestal's exposed height above the platform, in metres, "
        "instead of running the height curriculum; for replay and max-height tests",
    )
    return ("pedestal_height",)
