"""Reward and termination terms for `jumper.pedestal`.

The pedestal task keeps the tripod skeleton's velocity, upright and foot terms and
adds what mounting needs: progress of the root toward the pedestal top, a bonus
for actually standing on it, contact terms for the wall and rim, and a dense
shaping layer that allows a backward pitch, punishes a forward one, and closes the
two reward-hacking escapes -- a front-claw handstand and standing still far from
the wall. The height is read from the pedestal body rather than configured, so the
same terms work for every rung of the height curriculum.
"""

from __future__ import annotations

import inspect

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from scenes.pedestal import (
    PEDESTAL_HALF_HEIGHT,
    PEDESTAL_RADIUS,
    PEDESTAL_X,
    PLATFORM_TOP,
)

from ...common.constants import STAND_Z


def pedestal_climb(env, asset_cfg, pedestal_cfg) -> torch.Tensor:
    """Normalised root-height progress from the platform to the pedestal top.

    Zero with the robot standing on the platform and one once its root has risen
    by the exposed step height. Both ends are measured from the pedestal body's
    current placement, so a curriculum height changes the target without changing
    the term.
    """
    robot = env.scene[asset_cfg.name]
    pedestal = env.scene[pedestal_cfg.name]
    top = pedestal.data.root_link_pos_w[:, 2] + PEDESTAL_HALF_HEIGHT
    root = robot.data.root_link_pos_w[:, 2]
    start = PLATFORM_TOP + STAND_Z
    target = top + STAND_Z
    return torch.clamp((root - start) / (target - start).clamp_min(1e-6), 0.0, 1.0)


def pedestal_stand(env, asset_cfg, pedestal_cfg, margin: float = 0.04) -> torch.Tensor:
    """One while the root is at standing height on the pedestal top.

    The climb term saturates on height alone, which a rear-up can satisfy without
    the feet ever reaching the top. This term is the one that asks for the
    mounted pose; the margin is below the standing height so a slight settle does
    not turn the bonus off.
    """
    robot = env.scene[asset_cfg.name]
    pedestal = env.scene[pedestal_cfg.name]
    top = pedestal.data.root_link_pos_w[:, 2] + PEDESTAL_HALF_HEIGHT
    return (robot.data.root_link_pos_w[:, 2] > top + STAND_Z - margin).float()


def backward_lean(
    env,
    asset_cfg,
    tilt_deadband: float = 0.35,
    backward_deadband: float = 0.05,
) -> torch.Tensor:
    """Penalty for a body pitched/rolled too far, or driven backwards.

    The staircase task's term, kept here because a mount fails in the same two
    directions: backwards off the step and over on a tilt. The config carries the
    negative weight.
    """
    asset = env.scene[asset_cfg.name]
    gravity = asset.data.projected_gravity_b
    tilt = torch.clamp(
        torch.linalg.vector_norm(gravity[:, :2], dim=1) - tilt_deadband,
        min=0.0,
    )
    backward = torch.clamp(
        -asset.data.root_link_lin_vel_b[:, 0] - backward_deadband,
        min=0.0,
    )
    return tilt.square() + 2.0 * backward.square()


def forward_lean_penalty(
    env,
    asset_cfg,
    deadband: float = 0.10,
) -> torch.Tensor:
    """Penalty for pitching the nose toward the floor.

    ``projected_gravity_b`` is ``(0, 0, -1)`` level, so a nose-down pitch puts its
    x component positive and a nose-up (climbing) pitch puts it negative. Only the
    positive half is charged, and the deadband keeps ordinary walking pitch almost
    free while the square makes a steep nose-down pose expensive. The config
    carries the negative weight.
    """
    asset = env.scene[asset_cfg.name]
    gravity = asset.data.projected_gravity_b
    forward = torch.clamp(gravity[:, 0] - deadband, min=0.0)
    return forward.square()


def wall_hug_pitch(
    env,
    asset_cfg,
    wall_x: float = -0.30,
) -> torch.Tensor:
    """Reward a backward pitch (nose up) while the root is close to the wall.

    ``-projected_gravity_b[0]`` is zero with the body level and one with the nose
    straight up, so this is the pitch-up half of the attitude and nothing else.
    The proximity mask on root x keeps the bonus on the approach to the pedestal
    rather than paying it anywhere on the platform. Together with the reduced
    ``upright`` weight, this is what frees the robot to raise its front end and
    reach the rim.
    """
    asset = env.scene[asset_cfg.name]
    gravity = asset.data.projected_gravity_b
    backward_pitch = torch.clamp(-gravity[:, 0], min=0.0, max=1.0)
    near_wall = (asset.data.root_link_pos_w[:, 0] > wall_x).float()
    return backward_pitch * near_wall


def handstand_kill_switch(
    env,
    claw_cfg,
    rear_foot_cfg,
    platform_top: float = PLATFORM_TOP,
    claw_ground_z: float = 0.05,
    rear_foot_lift_z: float = 0.15,
) -> torch.Tensor:
    """One when the robot balances on both front claws with its rear legs up.

    This is the anti-handstand condition: **both** front claw tips within
    ``claw_ground_z`` of the platform surface **and** at least one middle/rear
    foot lifted more than ``rear_foot_lift_z`` above it. Heights are measured from
    the platform rather than in world coordinates, so the term survives the
    platform's fixed z. The config carries the extreme negative weight; returning
    1.0 is what makes that penalty fire.
    """
    claw_asset = env.scene[claw_cfg.name]
    rear_asset = env.scene[rear_foot_cfg.name]

    claw_z = (
        claw_asset.data.body_link_pos_w[:, claw_cfg.body_ids, 2] - platform_top
    )
    claws_down = (claw_z < claw_ground_z).all(dim=1)

    rear_z = (
        rear_asset.data.body_link_pos_w[:, rear_foot_cfg.body_ids, 2]
        - platform_top
    )
    rear_up = (rear_z > rear_foot_lift_z).any(dim=1)

    return (claws_down & rear_up).float()


class StagnationPenalty:
    """One while the root has stopped making the progress its phase needs.

    The old term watched only body-frame forward speed; a robot leaning on the
    wall could keep that near zero and still look "busy" to the reward, while a
    climbing robot was judged on an axis that points upward as it pitches. This
    term is phase-aware:

    - before the claws touch the pedestal (and before the root crosses the wall
      face) the robot must advance in world x;
    - once it has touched, it must advance in z (the climb).

    The contact latch is sticky for the episode, so a flickering claw sensor
    cannot switch the metric back and forth. The anchor moves only after a real
    advance, so slow but steady motion is not mistaken for a stall. Once the
    root is physically mounted the term is off: there is no height left to gain.
    """

    def __init__(self, cfg, env) -> None:
        del cfg
        self._last_metric = torch.zeros(env.num_envs, device=env.device)
        self._stall_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self._initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self._climb_phase = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._initialized[env_ids] = False
        self._stall_steps[env_ids] = 0
        self._last_metric[env_ids] = 0.0
        self._climb_phase[env_ids] = False

    def __call__(
        self,
        env,
        asset_cfg,
        pedestal_cfg,
        claw_sensor_name: str,
        min_progress_x: float = 0.01,
        min_progress_z: float = 0.01,
        patience_steps: int = 50,
        force_threshold: float = 0.0,
        radius_fraction: float = 0.5,
        height_tolerance: float = 0.05,
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        root = asset.data.root_link_pos_w

        sensor = env.scene[claw_sensor_name]
        contact = _contact_mask(sensor, force_threshold).any(dim=1)
        at_wall = root[:, 0] >= PEDESTAL_X - PEDESTAL_RADIUS
        previous_phase = self._climb_phase
        climb_phase = previous_phase | contact | at_wall
        self._climb_phase = climb_phase

        metric = torch.where(climb_phase, root[:, 2], root[:, 0])
        threshold = torch.where(
            climb_phase,
            torch.full_like(metric, min_progress_z),
            torch.full_like(metric, min_progress_x),
        )

        phase_changed = climb_phase != previous_phase
        self._initialized = self._initialized & ~phase_changed
        self._stall_steps = torch.where(
            phase_changed,
            torch.zeros_like(self._stall_steps),
            self._stall_steps,
        )
        self._last_metric = torch.where(
            phase_changed, metric, self._last_metric
        )

        fresh = ~self._initialized
        self._last_metric = torch.where(fresh, metric, self._last_metric)
        self._initialized = self._initialized | fresh

        advanced = (metric - self._last_metric) >= threshold
        self._stall_steps = torch.where(
            advanced,
            torch.zeros_like(self._stall_steps),
            self._stall_steps + 1,
        )
        # Only a real advance moves the anchor, so a sequence of small positive
        # steps still accumulates into progress.
        self._last_metric = torch.where(advanced, metric, self._last_metric)

        stalled = self._stall_steps >= patience_steps
        mounted = _mounted(
            env, asset, pedestal_cfg, radius_fraction, height_tolerance
        )
        return (stalled & ~mounted).float()


def below_platform(env, asset_cfg, min_height: float) -> torch.Tensor:
    """End the episode when the robot has left the platform.

    The platform is the only support at the start. Falling off it is not a
    policy learning a lower route; it is a reset that costs a whole episode of
    the only terrain this task has.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < min_height


def _contact_mask(sensor, force_threshold: float = 0.0) -> torch.Tensor:
    """Whether each primary has a non-zero contact force."""
    force = sensor.data.force
    assert force is not None, "contact sensor must expose force"
    return torch.linalg.vector_norm(force, dim=-1) > force_threshold


def _pedestal_top(env, pedestal_cfg) -> torch.Tensor:
    pedestal = env.scene[pedestal_cfg.name]
    return pedestal.data.root_link_pos_w[:, 2] + PEDESTAL_HALF_HEIGHT


def _pedestal_xy(env, pedestal_cfg) -> torch.Tensor:
    pedestal = env.scene[pedestal_cfg.name]
    return pedestal.data.root_link_pos_w[:, :2]


def _over_pedestal(
    env,
    asset,
    pedestal_cfg,
    radius_fraction: float = 0.5,
) -> torch.Tensor:
    """Whether the root's horizontal projection is inside the pedestal top.

    The check is against the pedestal body's own xy, not a hard-coded world x.
    ``radius_fraction`` below 1 keeps the root away from the rim, so a robot
    leaning on the side does not count as being on top.
    """
    root_xy = asset.data.root_link_pos_w[:, :2]
    distance = torch.linalg.vector_norm(
        root_xy - _pedestal_xy(env, pedestal_cfg), dim=1
    )
    return distance < PEDESTAL_RADIUS * radius_fraction


def _at_mount_height(
    env,
    asset,
    pedestal_cfg,
    height_tolerance: float = 0.05,
) -> torch.Tensor:
    """Whether the root is in the standing-height band above the pedestal top."""
    top = _pedestal_top(env, pedestal_cfg)
    error = asset.data.root_link_pos_w[:, 2] - (top + STAND_Z)
    return error.abs() < height_tolerance


def _mounted(
    env,
    asset,
    pedestal_cfg,
    radius_fraction: float = 0.5,
    height_tolerance: float = 0.05,
) -> torch.Tensor:
    """The physical success condition: over the pedestal and at its standing
    height, both read from the pedestal's current placement."""
    return _over_pedestal(
        env, asset, pedestal_cfg, radius_fraction
    ) & _at_mount_height(env, asset, pedestal_cfg, height_tolerance)


class GatedReward:
    """Wrap another reward term and multiply it by a root-x gate.

    The inner term keeps its own configuration and any class state (the posture
    term caches its stds, for example); this wrapper only evaluates it and
    applies a per-environment mask. It is used to switch the locomotion posture
    rewards off once the root is close to the wall, where the only useful
    attitude signal is ``wall_hug_pitch``.
    """

    def __init__(self, cfg, env) -> None:
        inner_cfg = cfg.params["inner_cfg"]
        for value in inner_cfg.params.values():
            if isinstance(value, SceneEntityCfg):
                value.resolve(env.scene)
        self._inner_cfg = inner_cfg
        if inspect.isclass(inner_cfg.func):
            self._inner = inner_cfg.func(cfg=inner_cfg, env=env)
        else:
            self._inner = inner_cfg.func

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        reset = getattr(self._inner, "reset", None)
        if reset is not None:
            reset(env_ids)

    def __call__(
        self,
        env,
        inner_cfg,
        asset_cfg,
        gate_x: float = -0.15,
    ) -> torch.Tensor:
        del inner_cfg
        asset = env.scene[asset_cfg.name]
        gate = (asset.data.root_link_pos_w[:, 0] <= gate_x).float()
        return self._inner(env, **self._inner_cfg.params) * gate


class ContactGatedReward:
    """Wrap a locomotion reward and pay it only before the claws touch.

    The latch is sticky for the episode, so a flickering contact sensor cannot
    switch the reward back on. Used for the tracking/gait/foot terms: once the
    robot is at the wall, those terms describe a walking task the robot is no
    longer doing, and they only add noise to the pull-up credit assignment.
    """

    def __init__(self, cfg, env) -> None:
        inner_cfg = cfg.params["inner_cfg"]
        for value in inner_cfg.params.values():
            if isinstance(value, SceneEntityCfg):
                value.resolve(env.scene)
        self._inner_cfg = inner_cfg
        if inspect.isclass(inner_cfg.func):
            self._inner = inner_cfg.func(cfg=inner_cfg, env=env)
        else:
            self._inner = inner_cfg.func
        self._contacted = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._contacted[env_ids] = False
        reset = getattr(self._inner, "reset", None)
        if reset is not None:
            reset(env_ids)

    def __call__(
        self,
        env,
        inner_cfg,
        claw_sensor_name: str,
        force_threshold: float = 0.0,
        active_before_contact: bool = True,
    ) -> torch.Tensor:
        del inner_cfg
        sensor = env.scene[claw_sensor_name]
        contact = _contact_mask(sensor, force_threshold).any(dim=1)
        self._contacted = self._contacted | contact
        gate = ~self._contacted if active_before_contact else self._contacted
        return self._inner(env, **self._inner_cfg.params) * gate.float()


class AttractorReward:
    """Gaussian pull of the claw tips toward the centre of the pedestal top.

    Active only after the first claw-pedestal contact and only until the robot
    is mounted. The two per-claw kernels are **multiplied**, not averaged: a
    single claw at the target cannot collect the term while the other arm is
    used as a counterweight. The root is deliberately not in the attractor; the
    jackpot and the vertical stagnation already cover the body.
    """

    def __init__(self, cfg, env) -> None:
        del cfg
        self._contacted = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._contacted[env_ids] = False

    def __call__(
        self,
        env,
        claw_cfg,
        pedestal_cfg,
        claw_sensor_name: str,
        scale: float = 0.25,
        force_threshold: float = 0.0,
        radius_fraction: float = 0.5,
        height_tolerance: float = 0.05,
    ) -> torch.Tensor:
        asset = env.scene[claw_cfg.name]
        sensor = env.scene[claw_sensor_name]
        contact = _contact_mask(sensor, force_threshold).any(dim=1)
        self._contacted = self._contacted | contact

        pedestal = env.scene[pedestal_cfg.name]
        top = _pedestal_top(env, pedestal_cfg)
        target = torch.cat(
            (pedestal.data.root_link_pos_w[:, :2], top.unsqueeze(1)), dim=1
        )
        claw_pos = asset.data.body_link_pos_w[:, claw_cfg.body_ids, :]
        distance = torch.linalg.vector_norm(
            claw_pos - target[:, None, :], dim=-1
        )
        pull = torch.exp(-distance / scale).prod(dim=1)

        before_mount = ~_mounted(
            env, asset, pedestal_cfg, radius_fraction, height_tolerance
        )
        return pull * self._contacted.float() * before_mount.float()


def claw_height_contact(
    env,
    claw_sensor_name: str,
    asset_cfg,
    pedestal_cfg,
    force_threshold: float = 0.0,
    radius_fraction: float = 0.5,
    height_tolerance: float = 0.05,
) -> torch.Tensor:
    """Reward the claws for touching the pedestal higher up its wall.

    The sensor is filtered to pedestal contacts, so a claw on the floor gets
    nothing even when another foot is at the same height. Height is measured
    above the platform, not in absolute world coordinates.

    The term is gated on the robot **not being mounted yet**: it is the pull-up
    currency, and it switches off only when the root is physically over the
    pedestal at standing height. A fixed root-x cutoff would switch it off just
    as the claws reach the wall, which is exactly the phase that needs it.
    """
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[claw_sensor_name]
    contact = _contact_mask(sensor, force_threshold)
    claw_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    height = (claw_z - PLATFORM_TOP).clamp_min(0.0)
    before_mount = ~_mounted(
        env, asset, pedestal_cfg, radius_fraction, height_tolerance
    )
    return (contact * height).sum(dim=1) * before_mount.float()


def claw_top_anchor(
    env,
    claw_sensor_name: str,
    asset_cfg,
    pedestal_cfg,
    margin: float = 0.04,
    force_threshold: float = 0.0,
    radius_fraction: float = 0.5,
    height_tolerance: float = 0.05,
) -> torch.Tensor:
    """Bonus for a claw contacting the pedestal at its top rim.

    ``claw_z`` is the J4 finger link, one ``STAND_Z`` above the claw tip, so
    the top-surface condition is ``pedestal_top + STAND_Z - margin``.

    Like ``claw_height_contact``, this is a pull-up term: it stays active through
    the wall contact and is zero only once the robot is actually mounted.
    """
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[claw_sensor_name]
    top = _pedestal_top(env, pedestal_cfg)
    contact = _contact_mask(sensor, force_threshold)
    claw_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    anchored = contact & (claw_z >= (top + STAND_Z - margin)[:, None])
    before_mount = ~_mounted(
        env, asset, pedestal_cfg, radius_fraction, height_tolerance
    )
    return anchored.float().all(dim=1) * before_mount.float()


def anchored_leg_lift(
    env,
    claw_sensor_name: str,
    feet_sensor_name: str,
    claw_cfg,
    foot_cfg,
    pedestal_cfg,
    anchor_margin: float = 0.04,
    lower_margin: float = 0.02,
    lower_contact_penalty: float = 1.0,
    force_threshold: float = 0.0,
) -> torch.Tensor:
    """Pull-up reward: lift the six feet only while a claw is anchored on top.

    The lift is the sum of each foot body's height above the platform. Any foot
    still touching below the top rim subtracts a fixed penalty, so a policy
    cannot collect the lift term by hanging from the arms with its legs down.
    This term is exactly zero before a claw reaches the top.
    """
    claw_asset = env.scene[claw_cfg.name]
    foot_asset = env.scene[foot_cfg.name]
    claw_sensor = env.scene[claw_sensor_name]
    feet_sensor = env.scene[feet_sensor_name]
    top = _pedestal_top(env, pedestal_cfg)

    claw_contact = _contact_mask(claw_sensor, force_threshold)
    claw_z = claw_asset.data.body_link_pos_w[:, claw_cfg.body_ids, 2]
    anchored = (claw_contact & (claw_z >= (top + STAND_Z - anchor_margin)[:, None])).all(dim=1)

    foot_z = foot_asset.data.body_link_pos_w[:, foot_cfg.body_ids, 2]
    foot_contact = _contact_mask(feet_sensor, force_threshold)
    lift = (foot_z - PLATFORM_TOP).clamp_min(0.0).sum(dim=1)
    lower_contact = (
        foot_contact & (foot_z < (top - lower_margin)[:, None])
    ).float().sum(dim=1)

    return anchored.float() * (lift - lower_contact_penalty * lower_contact)


class FullMountSuccess:
    """Jackpot: the root is physically over the pedestal at standing height.

    The acceptance condition is the body, not a foot sensor: the root's
    horizontal projection must be inside the pedestal top and its height must
    be in a band around ``pedestal_top + STAND_Z``. The condition has to hold
    for ``dwell_steps`` consecutive steps, so a transient rear-up or a leg waved
    at the rim cannot trigger it. It returns 1.0 once the dwell is met and keeps
    paying while the robot stays mounted.
    """

    def __init__(self, cfg, env) -> None:
        del cfg
        self._dwell = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._dwell[env_ids] = 0

    def __call__(
        self,
        env,
        asset_cfg,
        pedestal_cfg,
        radius_fraction: float = 0.5,
        height_tolerance: float = 0.05,
        dwell_steps: int = 10,
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        mounted = _mounted(
            env, asset, pedestal_cfg, radius_fraction, height_tolerance
        )
        self._dwell = torch.where(
            mounted,
            self._dwell + 1,
            torch.zeros_like(self._dwell),
        )
        return (self._dwell >= dwell_steps).float()

