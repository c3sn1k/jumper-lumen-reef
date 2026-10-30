"""Environment config for jumper.pedestal.

The tripod skeleton owns the walk, the tracking rewards and the foot terms. This
task owns the apparatus and the four things that make mounting it a different
skill:

- the two front grippers join the action and the observation, so the claws can
  hook the top edge (22 joints rather than the locomotion tasks' 20);
- a platform the robot starts on, and a cylinder on it whose exposed step is the
  task's difficulty;
- a height curriculum over that step, from 0.08 to 0.24 m;
- height-progress and mounted-pose rewards, a claw-to-rim cascade, and a
  termination for leaving the platform;
- a dense shaping layer that unlocks the backward pitch needed to climb, punishes
  a forward pitch, and removes the two escapes the old weights allowed: a
  front-claw handstand and standing still far from the wall.


The actor remains proprioceptive. The pedestal is found by contact, and the
critic can still use the skeleton's privileged foot/terrain terms. The task is
not a gait: the tripod skeleton and its command are the means, not the thing
being compared, so it is deliberately not in `test_task_parity.TASKS`.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import events as mdp_events
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from scenes.pedestal import (
    HEIGHT_MIN,
    PEDESTAL_HALF_HEIGHT,
    PLATFORM_TOP,
    pedestal_entity,
    platform_entity,
)

from ..common.constants import (
    ACTION_SCALE,
    FEET,
    FOOT_GEOMS,
    GAIT_JOINTS,
    GRIPPER_JOINTS,
)
from ..common.mdp.controls import controller_contract, load_controls
from ..tripod.env_cfg import env_cfg as _tripod_env_cfg
from .mdp.curriculum import PedestalHeightCurriculum
from .mdp.rewards import (
    AttractorReward,
    ContactGatedReward,
    FullMountSuccess,
    GatedReward,
    StagnationPenalty,
    anchored_leg_lift,
    below_platform,
    claw_height_contact,
    claw_top_anchor,
    forward_lean_penalty,
    handstand_kill_switch,
    pedestal_climb,
    pedestal_stand,
    wall_hug_pitch,
)

#: This task's operator controls. Generated from the dictionary by the controls
#: skill; the locomotion layout is the same because this task is driven the same
#: way, but it is this task's file and the contract is built from it.
CONTROLS = Path(__file__).parent / "controls.yaml"

#: How high a foot is asked to lift. The pedestal's tallest exposed step is
#: 0.24 m; 0.12 m is the largest the tripod leg can plausibly reach while the
#: front arms do the pulling.
FOOT_TARGET_HEIGHT = 0.12

#: The exposed step heights the curriculum visits, in metres above the platform.
HEIGHT_LEVELS = (0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24)
#: Control steps per rung. `common_step_counter` is per-environment control steps,
#: so this is ~167 PPO iterations at the tripod's 24 steps per iteration.
HEIGHT_DWELL_STEPS = 4000

#: Every joint the policy drives: the 20 gait joints plus the two claws. The
#: action order is this tuple; the export pairs joints by name, not by index.
ALL_JOINTS = tuple(GAIT_JOINTS) + tuple(GRIPPER_JOINTS)

#: The J4 claw links. Their subtree includes the finger, tip and grip insert
#: geoms, so one contact sensor covers every surface that can hook the rim.
CLAW_BODIES = ("LF_finger_link", "RF_finger_link")

#: The claw tips themselves, used by the handstand check. The finger link sits one
#: ``STAND_Z`` above the tip, so the tip is the body whose height can be compared
#: to the platform directly.
CLAW_TIP_BODIES = ("LF_finger_tip_link", "RF_finger_tip_link")

#: The middle and rear feet. The handstand check looks at these because a robot
#: balanced on its front claws has them in the air.
REAR_FOOT_BODIES = (
    "LM_foot_tip_link",
    "RM_foot_tip_link",
    "LR_foot_tip_link",
    "RR_foot_tip_link",
)


def env_cfg(
    asset: Path | None = None,
    play: bool = False,
    pedestal_height: float | None = None,
) -> ManagerBasedRlEnvCfg:
    """Build the pedestal environment on top of the tripod skeleton.

    `pedestal_height` fixes the exposed step and disables the height curriculum;
    `None` runs the curriculum during training. In replay/app mode `None` means
    the **final** 0.24 m step, because a deployed policy should be shown on the
    pedestal it is for; the fixed-height probe still overrides it.
    """
    if play and pedestal_height is None:
        pedestal_height = HEIGHT_LEVELS[-1]
    cfg = _tripod_env_cfg(asset=asset, play=play)
    # The upstream tripod keeps its foot-lift target (0.025 m) private and has no
    # terrain-scan switch. The pedestal needs a 0.12 m lift for the edge, so set
    # the two terms here instead of extending the shared tripod signature.
    for _name in ("foot_clearance", "foot_swing_height"):
        cfg.rewards[_name].params["target_height"] = FOOT_TARGET_HEIGHT

    # â”€â”€ 1. The claws are part of the skill --------------------------------
    # The skeleton restricts the action and the per-joint observations to the 20
    # gait joints. The two J4 joints are what can hook the pedestal edge, so they
    # are added here. `action.actuator_names` is the field that restricts the
    # action; `asset_cfg` on the action is silently ignored (see five_foot).
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.actuator_names = list(ALL_JOINTS)
    action.scale = {joint: ACTION_SCALE for joint in ALL_JOINTS}

    observed = SceneEntityCfg("robot", joint_names=list(ALL_JOINTS))
    observed_actuators = SceneEntityCfg("robot", actuator_names=list(ALL_JOINTS))
    for group in cfg.observations.values():
        for name in ("joint_pos", "joint_vel"):
            term = group.terms.get(name)
            if term is not None:
                term.params = dict(term.params or {}) | {"asset_cfg": observed}
        force = group.terms.get("actuator_force")
        if force is not None:
            force.params = dict(force.params or {}) | {"asset_cfg": observed_actuators}
    # The limits penalty should cover the two joints that are now being driven;
    # the pose reward stays on the 20 gait joints on purpose -- it must not charge
    # the claws for opening.
    cfg.rewards["dof_pos_limits"].params["asset_cfg"] = observed

    # â”€â”€ 2. The apparatus ---------------------------------------------------
    cfg.scene.entities["pedestal_platform"] = platform_entity()
    cfg.scene.entities["pedestal"] = pedestal_entity()

    # Claw contacts are filtered to the pedestal, so a claw resting on the
    # platform or the floor cannot collect the wall/top terms. The feet sensor
    # is unfiltered because the same six feet touch the platform, the pedestal
    # rim and (after a fall) the floor; the height of each foot tells them apart.
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ContactSensorCfg(
            name="claw_pedestal_contact",
            primary=ContactMatch(
                mode="subtree", pattern=CLAW_BODIES, entity="robot"
            ),
            secondary=ContactMatch(
                mode="body", pattern="pedestal", entity="pedestal"
            ),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
            history_length=cfg.decimation,
        ),
        ContactSensorCfg(
            name="feet_surface_contact",
            primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
            secondary=None,
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
            history_length=cfg.decimation,
        ),
    )
    cfg.events["reset_prop_platform"] = EventTermCfg(
        func=mdp_events.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("pedestal_platform"),
        },
    )
    cfg.events["reset_prop_pedestal"] = EventTermCfg(
        func=mdp_events.reset_root_state_uniform,
        mode="reset",
        params={
            # The pedestal's default z is HEIGHT_MIN above the platform. The
            # curriculum writes the part above it here.
            "pose_range": {"z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("pedestal"),
        },
    )

    # ── 3. Start on the platform, right in front of the pedestal ───────────
    # The skeleton's reset puts the robot on the ground and randomises x/y/yaw by
    # half a metre and a full turn. That is right for terrain variety and wrong
    # here: this task has one approach, from the platform's edge, and the policy
    # has to spend its time mounting rather than re-learning where the pedestal
    # is. `z` is an offset from the robot's standing height, so PLATFORM_TOP puts
    # the feet on the platform.
    #
    # **The spawn is deliberately against the wall.** The pedestal front face is
    # at x = 0.22 m world; the robot's front claws reach about 0.20 m ahead of the
    # root, so this range leaves 7-12 cm between the claw tips and the wall. The
    # old (-0.6, -0.4) start asked the policy to learn a 0.6-0.8 m approach on
    # top of the mount, and the stand-still local optimum won. The range stops at
    # -0.05 so every environment still has to make the small forward move that
    # `full_mount_success` and `stagnation_penalty` gate on.
    reset = cfg.events["reset_base"]
    reset.params["pose_range"] = {
        "x": (-0.10, -0.05),
        "y": (-0.03, 0.03),
        "z": (PLATFORM_TOP, PLATFORM_TOP),
        "yaw": (-0.03, 0.03),
    }


    robot = SceneEntityCfg("robot", body_names=("base_link",))
    pedestal = SceneEntityCfg("pedestal")
    claw_cfg = SceneEntityCfg("robot", body_names=CLAW_BODIES)
    claw_tip_cfg = SceneEntityCfg("robot", body_names=CLAW_TIP_BODIES)
    foot_cfg = SceneEntityCfg("robot", body_names=FEET)
    rear_foot_cfg = SceneEntityCfg("robot", body_names=REAR_FOOT_BODIES)

    # ── Dense cascade: wall → rim anchor → pull-up → full mount ──────────────
    # The root-height terms are retired (weight 0.0): they reward a rear-up
    # without a mount, which is the old reward-hacking path. The terms below are
    # ordered by how close they are to the actual skill, and the jackpot
    # dominates every other positive term in this environment.
    #
    # The two claw terms are the pull-up currency and their weights are raised
    # so they outbid the posture farm. They stay on until the root is physically
    # over the pedestal at standing height, which is also the jackpot condition.
    # `full_mount_success` reads the pedestal's own xy/z, not a foot sensor or a
    # hard-coded world x, and requires the condition to hold for a dwell window.
    cfg.rewards["pedestal_climb"] = RewardTermCfg(
        func=pedestal_climb,
        weight=0.0,
        params={"asset_cfg": robot, "pedestal_cfg": pedestal},
    )
    cfg.rewards["pedestal_stand"] = RewardTermCfg(
        func=pedestal_stand,
        weight=0.0,
        params={"asset_cfg": robot, "pedestal_cfg": pedestal},
    )
    cfg.rewards["claw_height_contact"] = RewardTermCfg(
        func=claw_height_contact,
        weight=8.0,
        params={
            "claw_sensor_name": "claw_pedestal_contact",
            "asset_cfg": claw_cfg,
            "pedestal_cfg": pedestal,
            "radius_fraction": 0.5,
            "height_tolerance": 0.05,
        },
    )
    cfg.rewards["claw_top_anchor"] = RewardTermCfg(
        func=claw_top_anchor,
        weight=15.0,
        params={
            "claw_sensor_name": "claw_pedestal_contact",
            "asset_cfg": claw_cfg,
            "pedestal_cfg": pedestal,
            "margin": 0.04,
            "radius_fraction": 0.5,
            "height_tolerance": 0.05,
        },
    )
    cfg.rewards["anchored_leg_lift"] = RewardTermCfg(
        func=anchored_leg_lift,
        weight=2.5,
        params={
            "claw_sensor_name": "claw_pedestal_contact",
            "feet_sensor_name": "feet_surface_contact",
            "claw_cfg": claw_cfg,
            "foot_cfg": foot_cfg,
            "pedestal_cfg": pedestal,
            "anchor_margin": 0.04,
            "lower_margin": 0.02,
            "lower_contact_penalty": 1.0,
        },
    )
    cfg.rewards["full_mount_success"] = RewardTermCfg(
        func=FullMountSuccess,
        weight=20.0,
        params={
            "asset_cfg": robot,
            "pedestal_cfg": pedestal,
            "radius_fraction": 0.5,
            "height_tolerance": 0.05,
            "dwell_steps": 10,
        },
    )
    # ── Contact phase machine ───────────────────────────────────────────────
    # Before the first claw-pedestal contact the task is still locomotion: the
    # tracking, gait and foot terms pay. The latch is sticky, so the switch is
    # one-way for the episode. After contact those terms stop: the robot is no
    # longer walking, and keeping them on would only add noise to the pull-up
    # credit assignment. The claw attractor is the dense post-contact term; the
    # jackpot and the guards stay unchanged.
    for _name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "tripod_gait",
        "stance_load",
        "foot_clearance",
        "foot_swing_height",
        "stance_ground_gap",
    ):
        _inner = cfg.rewards[_name]
        cfg.rewards[_name] = RewardTermCfg(
            func=ContactGatedReward,
            weight=_inner.weight,
            params={
                "inner_cfg": _inner,
                "claw_sensor_name": "claw_pedestal_contact",
                "force_threshold": 0.0,
                "active_before_contact": True,
            },
        )
    cfg.rewards["claw_attractor"] = RewardTermCfg(
        func=AttractorReward,
        weight=6.0,
        params={
            "claw_cfg": claw_tip_cfg,
            "pedestal_cfg": pedestal,
            "claw_sensor_name": "claw_pedestal_contact",
            "scale": 0.25,
            "force_threshold": 0.0,
            "radius_fraction": 0.5,
            "height_tolerance": 0.05,
        },
    )
    # ── Anti-handstand shaping ──────────────────────────────────────────────
    # The old weights assumed an upright mount. The robot instead learned to
    # pitch back, lean on the wall and farm the posture terms without climbing.
    # This layer removes that niche while keeping the pitch-up itself legal:
    #
    # - `upright` and `pose` are gated off once the root reaches the wall, where
    #   they only paid for standing still; `upright` also fought `wall_hug`;
    # - `wall_hug_pitch` remains the attitude term that permits the rear-up;
    # - `forward_lean_penalty` is the counterpart, so a nose-down pitch is not a
    #   second way to collect attitude reward;
    # - `handstand_kill_switch` is the extreme, conditional brake for balancing
    #   on both front claws with the rear feet in the air;
    # - `stagnation_penalty` is phase-aware: it requires x progress before the
    #   claws touch the pedestal and z progress after it, and stops once the
    #   root is mounted.
    cfg.rewards["upright"].weight = 0.5
    cfg.rewards["stance_load_stand"].weight = 0.0
    # Locomotion posture is only an approach reward. `upright` also fights the
    # pitch-up the climb needs (it wants the body level, `wall_hug_pitch` wants
    # it back), so both are switched off as soon as the root is at the wall.
    # The wrapper keeps the inner terms' own state and params intact.
    for _name in ("upright", "pose"):
        _inner = cfg.rewards[_name]
        cfg.rewards[_name] = RewardTermCfg(
            func=GatedReward,
            weight=_inner.weight,
            params={
                "inner_cfg": _inner,
                "asset_cfg": _inner.params["asset_cfg"],
                "gate_x": -0.15,
            },
        )
    cfg.rewards["forward_lean_penalty"] = RewardTermCfg(
        func=forward_lean_penalty,
        weight=-5.0,
        params={"asset_cfg": robot, "deadband": 0.10},
    )
    cfg.rewards["wall_hug_pitch"] = RewardTermCfg(
        func=wall_hug_pitch,
        weight=1.5,
        params={"asset_cfg": robot, "wall_x": -0.30},
    )
    cfg.rewards["handstand_kill_switch"] = RewardTermCfg(
        func=handstand_kill_switch,
        weight=-20.0,
        params={
            "claw_cfg": claw_tip_cfg,
            "rear_foot_cfg": rear_foot_cfg,
            "claw_ground_z": 0.05,
            "rear_foot_lift_z": 0.15,
        },
    )
    cfg.rewards["stagnation_penalty"] = RewardTermCfg(
        func=StagnationPenalty,
        weight=-5.0,
        params={
            "asset_cfg": robot,
            "pedestal_cfg": pedestal,
            "claw_sensor_name": "claw_pedestal_contact",
            "min_progress_x": 0.01,
            "min_progress_z": 0.01,
            "patience_steps": 50,
            "radius_fraction": 0.5,
            "height_tolerance": 0.05,
        },
    )
    cfg.terminations["below_platform"] = TerminationTermCfg(
        func=below_platform,
        params={"asset_cfg": robot, "min_height": PLATFORM_TOP + 0.04},
    )

    # â”€â”€ 4. Forward-only command, and a height curriculum -------------------
    # The operator command is the approach, not the skill: it asks the robot to
    # walk into the pedestal and stops there. There is no command curriculum --
    # the difficulty being widened is the step, not the speed.
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (0.0, 0.4)
    twist.ranges.lin_vel_y = (-0.05, 0.05)
    twist.ranges.ang_vel_z = (-0.1, 0.1)
    cfg.curriculum.pop("command", None)
    if pedestal_height is None:
        cfg.curriculum["pedestal_height"] = CurriculumTermCfg(
            func=PedestalHeightCurriculum,
            params={
                "levels": HEIGHT_LEVELS,
                "height_min": HEIGHT_MIN,
                "dwell_steps": HEIGHT_DWELL_STEPS,
                "event_name": "reset_prop_pedestal",
            },
        )
    else:
        # Fixed-height probe: no curriculum, one step for the whole run.
        if not 0.0 < pedestal_height <= 2.0 * PEDESTAL_HALF_HEIGHT:
            raise ValueError(
                f"pedestal_height must be in (0, {2.0 * PEDESTAL_HALF_HEIGHT:g}] m "
                f"(the cylinder body's height), got {pedestal_height:g}"
            )
        span = pedestal_height - HEIGHT_MIN
        cfg.events["reset_prop_pedestal"].params["pose_range"]["z"] = (span, span)
        cfg.curriculum.pop("pedestal_height", None)

    # â”€â”€ 5. Contact and constraint buffers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # The 22-joint pedestal puts both claws into the edge contact at once. The
    # skeleton's default buffers (128 contacts / 512 constraints) are sized for
    # the 20-joint locomotion tasks and overrun here: the solver drops contacts,
    # the PD has no support to push against, and the robot goes slack at frame
    # zero. In this mjlab the correct fields are:
    #
    #   cfg.sim.nconmax / njmax          warp's per-world buffers
    #   MjSpec.nconmax / njmax           the native compile, through scene.spec_fn
    #
    # `SceneCfg.mj_model_options` does not exist in this tree; the hasattr block
    # is kept as a no-op for a build that names it that way.
    cfg.sim.nconmax = 2000
    cfg.sim.njmax = 5000
    previous_spec_fn = cfg.scene.spec_fn

    def _contact_buffers(spec):
        if previous_spec_fn is not None:
            previous_spec_fn(spec)
        spec.nconmax = 2000
        spec.njmax = 5000

    cfg.scene.spec_fn = _contact_buffers
    if hasattr(cfg, "scene") and hasattr(cfg.scene, "mj_model_options"):
        cfg.scene.mj_model_options.nconmax = 2000
        cfg.scene.mj_model_options.njmax = 5000

    # The skeleton built the contract from tripod's controls file. Rebuild it from
    # this task's file, after the command ranges above, so an exported pedestal
    # contract carries the pedestal controls and the forward range.
    cfg.controller_contract = partial(controller_contract, load_controls(CONTROLS), cfg)
    return cfg


