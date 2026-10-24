"""Environment config for jumper.stairs.

The tripod skeleton owns the gait, the velocity reward and the command ladder.
This task owns the staircase apparatus and the three changes that make the same
gait climb it:

- the staircase is a physical prop, installed here so `--scene` is optional;
- the foot-lift target is raised from the tripod's 0.025 m to 0.06 m, matching
  the tallest rise; the rises are 0.03-0.06 m;
- a height-progress reward and a backward/tilt penalty are added, and the
  foot-clearance penalties are strengthened so the feet clear the vertical edge
  instead of catching it.

The actor remains proprioceptive only; the staircase is found by contact. That
is what the deployed controller can run.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from scenes.temple_stairs import TOP_HEIGHT, stairs_entity

from ..tripod.env_cfg import env_cfg as _tripod_env_cfg
from .mdp.rewards import backward_lean, climb_height

#: Foot-lift target this task asks for, in metres. The tallest rise is 0.06.
FOOT_TARGET_HEIGHT = 0.06
#: Root-height progress that counts as climbed, measured from the standing root.
START_HEIGHT = 0.10
TARGET_HEIGHT = TOP_HEIGHT - 0.05


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the stairs environment on top of the tripod skeleton."""
    from mjlab.envs.mdp import events as mdp_events

    cfg = _tripod_env_cfg(
        asset=asset,
        play=play,
        terrain_scan=False,
        foot_target_height=FOOT_TARGET_HEIGHT,
    )

    # The staircase is a fixed-base prop; mjlab wraps it in a mocap frame and
    # needs this reset event to place one per environment. This is the same
    # event `scenes/registry._place_props` installs for a scene prop.
    cfg.scene.entities["stairs"] = stairs_entity()
    cfg.events["reset_prop_stairs"] = EventTermCfg(
        func=mdp_events.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("stairs"),
        },
    )

    robot = SceneEntityCfg("robot", body_names=("base_link",))
    cfg.rewards["climb_height"] = RewardTermCfg(
        func=climb_height,
        weight=1.5,
        params={
            "asset_cfg": robot,
            "start_height": START_HEIGHT,
            "target_height": TARGET_HEIGHT,
        },
    )
    cfg.rewards["backward_lean"] = RewardTermCfg(
        func=backward_lean,
        weight=-2.0,
        params={
            "asset_cfg": robot,
            "tilt_deadband": 0.35,
            "backward_deadband": 0.05,
        },
    )

    # The skeleton already set both targets to FOOT_TARGET_HEIGHT. These are
    # penalties around that target, so a stronger weight asks the feet to reach
    # it more reliably without changing the target itself.
    cfg.rewards["foot_clearance"].weight = -3.0
    cfg.rewards["foot_swing_height"].weight = -0.5

    return cfg
